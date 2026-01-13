"""Utilities for computing text embeddings."""

import os
import time
import glob
import gc
import random
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from tqdm.auto import tqdm
import tiktoken
import torch
import openai
from sentence_transformers import SentenceTransformer

from .utils import filter_invalid_texts


# Use environment variable for cache dir if set, otherwise use default
CACHE_DIR = os.getenv("EMB_CACHE_DIR") or os.path.join(Path(__file__).parent.parent, "emb_cache")


# -----------------------------
# Token-aware batching + truncation
# -----------------------------
def _truncate_and_count_tokens(
    text: str,
    enc,
    per_item_cap: int,
) -> Tuple[str, int]:
    """Strip, tokenize, truncate to per_item_cap, return (possibly truncated text, token_count)."""
    text = text.strip()
    toks = enc.encode(text)
    if len(toks) > per_item_cap:
        toks = toks[:per_item_cap]
        text = enc.decode(toks)
    return text, len(toks)


def _make_token_batches(
    texts: List[str],
    token_budget: int = 60_000,
    per_item_cap: int = 8192,
    encoding_name: str = "cl100k_base",
) -> List[List[str]]:
    """
    Build batches such that total tokens per request stays under token_budget.
    Also truncates each text to per_item_cap tokens.

    Note: token_budget should be set well below your org's TPM if using concurrency.
    """
    enc = tiktoken.get_encoding(encoding_name)

    batches: List[List[str]] = []
    cur: List[str] = []
    cur_tokens = 0

    for raw in texts:
        # Defensive: raw should already be valid from filter_invalid_texts, but keep robust.
        if raw is None:
            continue

        text, n = _truncate_and_count_tokens(raw, enc, per_item_cap)

        # If adding would exceed token_budget, flush current batch.
        if cur and (cur_tokens + n > token_budget):
            batches.append(cur)
            cur = []
            cur_tokens = 0

        cur.append(text)
        cur_tokens += n

    if cur:
        batches.append(cur)

    return batches


# -----------------------------
# OpenAI embedding call with robust retries
# -----------------------------
def _embed_batch_openai(
    batch: List[str],
    model: str,
    client,
    max_retries: int = 6,
    backoff_base_s: float = 0.8,
    backoff_factor: float = 2.0,
    timeout: float = 60.0,
) -> List[List[float]]:
    """
    Embed a batch of strings using OpenAI embeddings API.
    Assumes texts are already truncated appropriately.
    Retries on rate limits/timeouts with exponential backoff + jitter.
    """
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                input=batch,
                model=model,
                timeout=timeout,
            )
            return [d.embedding for d in resp.data]

        except (openai.RateLimitError, openai.APITimeoutError) as e:
            last_err = e
            sleep_s = backoff_base_s * (backoff_factor ** attempt)
            sleep_s *= (0.8 + 0.4 * random.random())

            if attempt < max_retries - 1:
                if attempt >= 0:
                    print(f"API error: {type(e).__name__}: {e}; retrying in {sleep_s:.2f}s "
                          f"({attempt + 1}/{max_retries})")
                time.sleep(sleep_s)
            else:
                raise

    raise last_err if last_err else RuntimeError("Unknown embedding error")


# -----------------------------
# Cache utilities
# -----------------------------
def load_embedding_cache(cache_name: str) -> dict:
    """Load cached embeddings from chunked files."""
    if not cache_name:
        return {}

    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return {}

    text2embedding = {}
    chunk_files = sorted(glob.glob(f"{cache_dir}/chunk_*.npy"))

    for chunk_file in tqdm(chunk_files, desc="Loading embedding chunks"):
        chunk_data = np.load(chunk_file, allow_pickle=True)
        for text, emb in chunk_data:
            text2embedding[text] = emb

    return text2embedding


def _get_next_chunk_index(cache_name: str) -> int:
    """Determine the next available chunk index for a cache."""
    if not cache_name:
        return 0

    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return 0

    chunk_files = glob.glob(f"{cache_dir}/chunk_*.npy")
    if not chunk_files:
        return 0

    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in chunk_files]
    return max(indices) + 1


def _save_embedding_chunk(cache_name: str, chunk_embeddings: dict, chunk_idx: int) -> int:
    """Save a chunk of embeddings to disk."""
    if not cache_name or not chunk_embeddings:
        return chunk_idx

    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)

    chunk_path = f"{cache_dir}/chunk_{chunk_idx:03d}.npy"
    chunk_items = list(chunk_embeddings.items())
    np.save(chunk_path, np.array(chunk_items, dtype=object))

    return chunk_idx + 1


# -----------------------------
# Public API: OpenAI embeddings
# -----------------------------
def get_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    # Kept for backward compatibility (ignored for OpenAI path)
    batch_size: int = 256,
    # Concurrency: start at 1; try 2 only after confirming no 429 churn
    n_workers: int = 1,
    cache_name: Optional[str] = None,
    show_progress: bool = True,
    chunk_size: int = 50_000,
    timeout: float = 60.0,
    # Token-aware batching controls
    token_budget: int = 60_000,
    per_item_cap: int = 8192,
) -> Dict[str, np.ndarray]:
    """Get embeddings using OpenAI API with token-aware batching and chunked caching."""
    texts = filter_invalid_texts(texts)

    # Load cache and filter already-embedded texts
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [t for t in texts if t not in text2embedding]

    if not texts_to_embed:
        return text2embedding

    from .llm_api import get_client
    client = get_client()

    # Process in chunks to bound memory and to write intermediate progress to disk
    next_chunk_idx = _get_next_chunk_index(cache_name)

    chunk_ranges = [(i, min(i + chunk_size, len(texts_to_embed)))
                    for i in range(0, len(texts_to_embed), chunk_size)]

    chunk_iter = tqdm(chunk_ranges, desc="Processing chunks", total=len(chunk_ranges)) if show_progress else chunk_ranges

    for chunk_start, chunk_end in chunk_iter:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings: Dict[str, np.ndarray] = {}

        batches = _make_token_batches(
            chunk_texts,
            token_budget=token_budget,
            per_item_cap=per_item_cap,
            encoding_name="cl100k_base",
        )

        enc = tiktoken.get_encoding("cl100k_base")
        truncated_to_original = {}
        for original_text in chunk_texts:
            text_stripped = original_text.strip()
            toks = enc.encode(text_stripped)
            if len(toks) > per_item_cap:
                toks = toks[:per_item_cap]
                truncated = enc.decode(toks)
                truncated_to_original[truncated] = original_text
            else:
                truncated_to_original[text_stripped] = original_text
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_batch = {
                executor.submit(_embed_batch_openai, batch, model, client, timeout=timeout): batch
                for batch in batches
            }

            fut_iter = concurrent.futures.as_completed(future_to_batch)
            if show_progress:
                fut_iter = tqdm(fut_iter, total=len(batches), desc=f"Chunk {next_chunk_idx}")

            for fut in fut_iter:
                batch = future_to_batch[fut]
                embs = fut.result()
                for truncated_text, emb in zip(batch, embs):
                    original_text = truncated_to_original.get(truncated_text, truncated_text)
                    chunk_embeddings[original_text] = emb
                    text2embedding[original_text] = emb

        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)

    return text2embedding


# -----------------------------
# Public API: Local embeddings
# -----------------------------
def get_local_embeddings(
    texts: List[str],
    model: str = "nomic-ai/modernbert-embed-base",
    batch_size: int = 128,
    show_progress: bool = True,
    cache_name: Optional[str] = None,
    chunk_size: int = 50_000,
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, np.ndarray]:
    """Get embeddings using local SentenceTransformer model with chunked caching."""
    texts = filter_invalid_texts(texts)

    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [t for t in texts if t not in text2embedding]
    if not texts_to_embed:
        return text2embedding

    transformer_model = SentenceTransformer(model, device=device)

    next_chunk_idx = _get_next_chunk_index(cache_name)

    chunk_ranges = [(i, min(i + chunk_size, len(texts_to_embed)))
                    for i in range(0, len(texts_to_embed), chunk_size)]
    chunk_iter = tqdm(chunk_ranges, desc="Processing chunks", total=len(chunk_ranges)) if show_progress else chunk_ranges

    for chunk_start, chunk_end in chunk_iter:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings: Dict[str, np.ndarray] = {}

        batch_iter = range(0, len(chunk_texts), batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc=f"Chunk {next_chunk_idx}")

        for i in batch_iter:
            batch = chunk_texts[i:i + batch_size]

            if "nomic-ai" in model:
                prefixed = ["clustering: " + t for t in batch]
            elif "instructor" in model:
                prefixed = [["Represent the text for classification: ", t] for t in batch]
            else:
                prefixed = batch

            batch_embs = transformer_model.encode(prefixed, batch_size=batch_size)

            for text, emb in zip(batch, batch_embs):
                chunk_embeddings[text] = emb
                text2embedding[text] = emb

        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)

    del transformer_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text2embedding
