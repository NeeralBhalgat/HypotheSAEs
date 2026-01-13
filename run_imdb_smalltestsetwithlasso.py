"""Minimal cost HypotheSAEs on tiny IMDB subset using embedding similarity annotation and LASSO selection."""

import os
import sys
import numpy as np
import pandas as pd

# Add path for hypothesaes
sys.path.insert(0, os.getcwd())

# ============================================================================
# CONFIGURATION: Toggle between annotation methods
# ============================================================================
# Options:
#   "embedding" - Use embedding similarity (no API cost, fast)
#   "local_llm" - Use local LLM like Qwen (no API cost, requires GPU)
ANNOTATION_METHOD = "embedding"  # Change to "local_llm" to use local models

# Local LLM settings (only used if ANNOTATION_METHOD == "local_llm")
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Options: "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"

# Interpreter model (for generating hypotheses)
# Options: "gpt-4.1-mini" (OpenAI API), or local Qwen model
INTERPRETER_MODEL = LOCAL_LLM_MODEL  # Using Qwen for interpretation

# ============================================================================

# Set your OpenAI API key: os.environ['OPENAI_KEY_SAE'] = "your-api-key-here"

print("=" * 80)
print("QWEN TEST: HypotheSAEs with Qwen interpreter (1-minute run)")
print("=" * 80)
print("Using tiny subset for fast Qwen testing")
print(f"Annotation: {ANNOTATION_METHOD} (no API cost)")
if ANNOTATION_METHOD == "local_llm":
    print(f"Local LLM: {LOCAL_LLM_MODEL}")
print(f"Interpreter: {INTERPRETER_MODEL} (Qwen)")
print("Selection: LASSO (L1 regression)")
print()

print("STEP 1: Imports")
print("-" * 80)
from hypothesaes.quickstart import train_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_openai_embeddings

print("[OK] Imports successful!")
print()

print("STEP 2: Load Tiny Subset")
print("-" * 80)
current_dir = os.getcwd()
prefix = "./" if not current_dir.endswith("notebooks") else "../"
base_dir = os.path.join(prefix, "demo_data")

train_df = pd.read_json(os.path.join(base_dir, "imdb-demo-train-20K.json"), lines=True)
val_df = pd.read_json(os.path.join(base_dir, "imdb-demo-val-2K.json"), lines=True)
holdout_df = pd.read_json(os.path.join(base_dir, "imdb-demo-holdout-2K.json"), lines=True)

# TINY subset for 1-minute Qwen run
TRAIN_SIZE = 15  # Very small for fast Qwen run
VAL_SIZE = 5
HOLDOUT_SIZE = 5

texts = train_df['text'].head(TRAIN_SIZE).tolist()
labels = train_df['stars'].head(TRAIN_SIZE).values
val_texts = val_df['text'].head(VAL_SIZE).tolist()
holdout_texts = holdout_df['text'].head(HOLDOUT_SIZE).tolist()
holdout_labels = holdout_df['stars'].head(HOLDOUT_SIZE).values

print(f"[OK] Loaded {len(texts)} training, {len(val_texts)} validation, {len(holdout_texts)} holdout")
print(f"  Labels: {labels.min()}-{labels.max()}")
print()

print("STEP 3: Compute Embeddings (OpenAI)")
print("-" * 80)
EMBEDDER = "text-embedding-3-small"  # Cheapest embedding model
CACHE_NAME = f"imdb_minimal_embedding_lasso_{EMBEDDER}"  # Different cache to avoid conflicts

print(f"Using {EMBEDDER}...")
print(f"Embedding {len(texts) + len(val_texts)} texts")
print(f"Estimated cost: ~${0.00002 * (len(texts) + len(val_texts)):.4f}")
text2embedding = get_openai_embeddings(
    texts + val_texts,
    model=EMBEDDER,
    cache_name=CACHE_NAME,
    n_workers=1  # Single worker to avoid rate limits
)

# Filter to only texts that have embeddings
train_embeddings_list = []
texts_filtered = []
labels_filtered = []

for text, label in zip(texts, labels):
    if text in text2embedding:
        train_embeddings_list.append(text2embedding[text])
        texts_filtered.append(text)
        labels_filtered.append(label)

val_embeddings_list = []
val_texts_filtered = []

for text in val_texts:
    if text in text2embedding:
        val_embeddings_list.append(text2embedding[text])
        val_texts_filtered.append(text)

if len(texts_filtered) < len(texts):
    print(f"Warning: {len(texts) - len(texts_filtered)} texts were filtered out")
if len(val_texts_filtered) < len(val_texts):
    print(f"Warning: {len(val_texts) - len(val_texts_filtered)} validation texts were filtered out")

texts = texts_filtered
val_texts = val_texts_filtered
labels = np.array(labels_filtered)

train_embeddings = np.stack(train_embeddings_list)
val_embeddings = np.stack(val_embeddings_list)

print(f"[OK] Embeddings shape: {train_embeddings.shape}")
print()

print("STEP 4: Train SAE (Tiny for Speed)")
print("-" * 80)
checkpoint_dir = os.path.join(prefix, "checkpoints", CACHE_NAME)

# Small SAE for minimal dataset
M, K = 16, 2  # Very tiny SAE for 1-minute run
print(f"Training SAE: M={M}, K={K} (very tiny for speed)...")
sae = train_sae(
    embeddings=train_embeddings,
    val_embeddings=val_embeddings,
    M=M,
    K=K,
    checkpoint_dir=checkpoint_dir,
    n_epochs=5,  # Very few epochs for speed
    batch_size=32,
    show_progress=True
)

print(f"[OK] SAE trained!")
print()

print("STEP 5: Generate Hypotheses (LASSO Selection)")
print("-" * 80)
print(f"Using {INTERPRETER_MODEL} for interpretation")
print("Using LASSO for neuron selection (L1 regression)")
print("Interpreting only 2 neurons for fast Qwen run")
print()

TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are movie reviews from IMDB.
Features should describe a specific aspect of the review. For example:
- "mentions the director or specific filmmaking techniques"
- "criticizes the plot or storyline as confusing or poorly written"
- "praises the acting performance of specific actors"
"""

# Use LASSO selection and minimal neurons to reduce API calls
results = generate_hypotheses(
    texts=texts,
    labels=labels,
    embeddings=train_embeddings,
    sae=sae,
    cache_name=CACHE_NAME,
    selection_method="lasso",  # CHANGED: Use LASSO instead of correlation
    n_selected_neurons=2,  # Minimal neurons for fast Qwen run
    n_candidate_interpretations=1,  # Skip scoring to save cost
    n_scoring_examples=0,  # Skip scoring step entirely
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    interpreter_model=INTERPRETER_MODEL,  # Use Qwen for interpretation
    annotator_model=LOCAL_LLM_MODEL if ANNOTATION_METHOD == "local_llm" else "gpt-4.1-mini",  # Only used if annotation_method != "embedding"
    n_workers_annotation=1,  # Minimal parallelism to avoid rate limits
    n_workers_interpretation=1,  # Single worker for Qwen
    n_examples_for_interpretation=5,  # Very few examples for speed
    max_words_per_example=256,  # Standard length
)

print(f"[OK] Generated {len(results)} hypotheses")
print("\nHypotheses:")
pd.set_option('display.max_colwidth', 100)
print(results[['neuron_idx', 'target_lasso', 'interpretation']])
pd.reset_option('display.max_colwidth')
print()

print("STEP 6: Evaluate on Holdout")
print("-" * 80)
print(f"Evaluating on {len(holdout_texts)} holdout samples...")
if ANNOTATION_METHOD == "embedding":
    print(f"Using embedding similarity annotation (no API cost!)")
    print(f"Similarity threshold: 0.7")
    # Get embeddings for holdout texts (reuse same embedding model)
    print("Computing embeddings for holdout texts...")
    holdout_text2embedding = get_openai_embeddings(
        holdout_texts,
        model=EMBEDDER,
        cache_name=CACHE_NAME,
        n_workers=1  # Single worker to avoid rate limits
    )
else:
    print(f"Using local LLM annotation: {LOCAL_LLM_MODEL} (no API cost!)")
    holdout_text2embedding = None

if ANNOTATION_METHOD == "embedding":
    metrics, evaluation_df = evaluate_hypotheses(
        hypotheses_df=results,
        texts=holdout_texts,
        labels=holdout_labels,
        cache_name=CACHE_NAME,
        annotation_method="embedding",  # Use embedding similarity
        embedding_model=EMBEDDER,  # Use same embedding model
        similarity_threshold=0.7,  # Default threshold
        use_local_embeddings=False,  # Using OpenAI embeddings
        text2embedding=holdout_text2embedding,  # Reuse computed embeddings
    )
else:
    # Use local LLM for annotation
    metrics, evaluation_df = evaluate_hypotheses(
        hypotheses_df=results,
        texts=holdout_texts,
        labels=holdout_labels,
        cache_name=CACHE_NAME,
        annotation_method="llm",  # Use LLM annotation
        annotator_model=LOCAL_LLM_MODEL,  # Use local LLM
        n_workers_annotation=1,  # Minimal parallelism
        max_words_per_example=256,
    )

print(f"[OK] Evaluation complete!")
print("\nHoldout Set Metrics:")
print(f"R² Score: {metrics['r2']:.3f}")
print(f"Significant hypotheses: {metrics['Significant'][0]}/{metrics['Significant'][1]} (p < {metrics['Significant'][2]:.3e})")
print("\nHypothesis Results:")
pd.set_option('display.max_colwidth', 100)
print(evaluation_df[['hypothesis', 'separation_score', 'regression_coef', 'feature_prevalence']].round(3))
pd.reset_option('display.max_colwidth')
print()

print("=" * 80)
print("COMPLETE! Qwen test workflow executed.")
print("=" * 80)
print("Cost breakdown:")
embedding_cost = 0.00002 * (len(texts) + len(val_texts) + len(holdout_texts))  # Tiny set

# Interpretation cost depends on model
if INTERPRETER_MODEL.startswith("gpt"):
    interpretation_cost = 0.001  # GPT-4.1-mini for 2 neurons (very cheap)
else:
    interpretation_cost = 0.0  # Local LLM (free)

# Annotation cost depends on method
if ANNOTATION_METHOD == "embedding":
    annotation_cost = 0.0  # FREE! Using embedding similarity
    annotation_desc = "embedding similarity"
else:
    annotation_cost = 0.0  # FREE! Using local LLM
    annotation_desc = f"local LLM ({LOCAL_LLM_MODEL})"

total_cost = embedding_cost + interpretation_cost + annotation_cost

print(f"  - Embeddings ({len(texts) + len(val_texts) + len(holdout_texts):,} texts): ~${embedding_cost:.4f}")
print(f"  - Interpretations (2 neurons, {INTERPRETER_MODEL}): ~${interpretation_cost:.4f} (FREE with Qwen!)")
print(f"  - Annotations ({annotation_desc}): ${annotation_cost:.2f} (FREE!)")
print(f"  Total: ~${total_cost:.4f} (Qwen test - should run in ~1 minute)")
