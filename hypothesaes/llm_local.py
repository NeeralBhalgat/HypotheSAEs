"""Local LLM utilities for HypotheSAEs."""
import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
torch.set_float32_matmul_precision("high")

from typing import List, Optional
from functools import lru_cache
from tqdm.auto import tqdm

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Create dummy types for type hints
    LLM = None
    SamplingParams = None

import time

_LOCAL_ENGINES: dict[str, "LLM"] = {}

@lru_cache(maxsize=256) # Cache models that we've already checked
def hf_model_exists(repo_id: str) -> bool:
    try:
        HfApi().model_info(repo_id, timeout=3)
        return True                              
    except RepositoryNotFoundError:
        return False
    except HTTPError as e:
        return e.response is not None and e.response.status_code in {401, 403}

def is_local_model(model: str) -> bool:
    """Check if model is a local model (vLLM, transformers, or ollama)."""
    # Ollama models
    if model.startswith("ollama:"):
        return True
    # Cached vLLM engines
    if model in _LOCAL_ENGINES:
        return True
    # Check if it's a HuggingFace model (works with both vLLM and transformers)
    if hf_model_exists(model):
        return True
    return False

def _sleep_all_except(active_model: Optional[str] = None) -> None:
    """Put every cached vLLM engine *except* `active` to sleep."""
    if not VLLM_AVAILABLE:
        return
    for name, engine in _LOCAL_ENGINES.items():
        if name == active_model:
            continue
        if engine.llm_engine.is_sleeping():
            continue
        engine.llm_engine.reset_prefix_cache()
        engine.sleep(level=2)

def get_vllm_engine(model: str, **kwargs) -> "LLM":
    """
    Return a vLLM engine for `model`.

    * If the engine is already cached, sleep the others and wake it.
    * If it is not cached, sleep every other engine first so the GPU
      is empty, then load the new model.
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vllm is not installed. Install it with: pip install vllm")
    
    engine = _LOCAL_ENGINES.get(model)

    if engine is None:
        _sleep_all_except(active_model=None) # free GPU before allocating

        gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.85)
        engine = LLM(model=model, enable_sleep_mode=True, gpu_memory_utilization=gpu_memory_utilization, **kwargs)
        _LOCAL_ENGINES[model] = engine
    else:
        _sleep_all_except(active_model=model)
        if engine.llm_engine.is_sleeping(): 
            engine.wake_up()
            engine.llm_engine.reset_prefix_cache()

    return engine

def shutdown_all_vllm_engines() -> None:
    """Shut down and clear any cached vLLM engines to release GPU resources."""
    if not VLLM_AVAILABLE:
        return
    global _LOCAL_ENGINES
    for name, engine in list(_LOCAL_ENGINES.items()):
        try:
            engine.llm_engine.engine_core.shutdown()
        except Exception as exc:
            print(f"Warning: failed to shut down vLLM engine '{name}': {exc}")
    _LOCAL_ENGINES.clear()

def get_local_completions(
    prompts: List[str],
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 128,
    show_progress: bool = True,
    tokenizer_kwargs: Optional[dict] = {},
    llm_sampling_kwargs: Optional[dict] = {},
) -> List[str]:
    """Generate completions using vLLM, ollama, or transformers."""
    # Check if using ollama (model name starts with "ollama:")
    if model.startswith("ollama:"):
        return _get_ollama_completions(prompts, model[7:], max_tokens, show_progress, llm_sampling_kwargs)
    
    # Try vLLM first
    if VLLM_AVAILABLE:
        try:
            engine = get_vllm_engine(model)
            tokenizer = engine.get_tokenizer()

            if getattr(tokenizer, "chat_template", None) is not None:
                messages_lists = [[{"role": "user", "content": p}] for p in prompts]
                enable_thinking = tokenizer_kwargs.pop("enable_thinking", False) # Default to False so users don't get unexpected output
                prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking, **tokenizer_kwargs)
                           for messages in messages_lists]

            sampling_params = SamplingParams(max_tokens=max_tokens, **llm_sampling_kwargs)
            outputs = engine.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=show_progress,
            )

            completions = [str(out.outputs[0].text) for out in outputs]
            return completions
        except Exception as e:
            print(f"vLLM failed: {e}, falling back to transformers")
    
    # Fallback to transformers (works on Windows/CPU)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        raise ImportError("Neither vllm nor transformers is installed. Install with: pip install transformers")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True
    )
    if device.type == "cpu":
        model_obj = model_obj.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    completions = []
    iterator = tqdm(prompts, desc="Generating", disable=not show_progress) if show_progress else prompts
    
    for prompt in iterator:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        temperature = llm_sampling_kwargs.get("temperature", 0.7)
        do_sample = temperature > 0
        
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(generated_text.strip())
    
    return completions

def _get_ollama_completions(
    prompts: List[str],
    model: str,
    max_tokens: int,
    show_progress: bool,
    llm_sampling_kwargs: Optional[dict] = {},
) -> List[str]:
    """Generate completions using ollama API."""
    try:
        import ollama
    except ImportError:
        raise ImportError("ollama not installed. Install with: pip install ollama")
    
    completions = []
    iterator = tqdm(prompts, desc="Generating (ollama)", disable=not show_progress) if show_progress else prompts
    
    temperature = llm_sampling_kwargs.get("temperature", 0.7)
    
    for prompt in iterator:
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            )
            completions.append(response.get("response", "").strip())
        except Exception as e:
            if "Failed to connect" in str(e) or "Connection" in str(e):
                raise ConnectionError(
                    f"Ollama is not running. Please:\n"
                    f"  1. Install ollama from https://ollama.ai\n"
                    f"  2. Start it: ollama serve (or it may auto-start)\n"
                    f"  3. Pull the model: ollama pull {model}\n"
                    f"Original error: {e}"
                )
            print(f"Ollama error for prompt: {e}")
            completions.append("")
    
    return completions
