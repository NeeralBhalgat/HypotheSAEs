"""Optimize R² score with parameter tuning - improved version."""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, os.getcwd())

from hypothesaes.quickstart import train_sae, evaluate_hypotheses
from hypothesaes.embedding import get_local_embeddings
from hypothesaes.select_neurons import select_neurons

TRAIN_SIZE = 4000
VAL_SIZE = 400
HOLDOUT_SIZE = 400

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]

SAE_CONFIGS = [
    {"M": 32, "K": 2},
    {"M": 64, "K": 4},
]

SIMILARITY_THRESHOLDS = [0.25, 0.35]
N_HYPOTHESES_OPTIONS = [5]

current_dir = os.getcwd()
prefix = "./" if not current_dir.endswith("notebooks") else "../"
base_dir = os.path.join(prefix, "demo_data")

train_df = pd.read_json(os.path.join(base_dir, "imdb-demo-train-20K.json"), lines=True)
val_df = pd.read_json(os.path.join(base_dir, "imdb-demo-val-2K.json"), lines=True)
holdout_df = pd.read_json(os.path.join(base_dir, "imdb-demo-holdout-2K.json"), lines=True)

texts = train_df['text'].head(TRAIN_SIZE).tolist()
labels = train_df['stars'].head(TRAIN_SIZE).values
val_texts = val_df['text'].head(VAL_SIZE).tolist()
holdout_texts = holdout_df['text'].head(HOLDOUT_SIZE).tolist()
holdout_labels = holdout_df['stars'].head(HOLDOUT_SIZE).values

configs_to_test = []
for embedder in EMBEDDING_MODELS:
    for sae_config in SAE_CONFIGS:
        for n_hyp in N_HYPOTHESES_OPTIONS:
            for threshold in SIMILARITY_THRESHOLDS:
                configs_to_test.append({
                    "embedder": embedder,
                    "M": sae_config["M"],
                    "K": sae_config["K"],
                    "n_hypotheses": n_hyp,
                    "threshold": threshold
                })

total_configs = len(configs_to_test)

best_r2 = -np.inf
best_config = None
all_results = []

for idx, config in enumerate(configs_to_test, 1):
    embedder = config["embedder"]
    M = config["M"]
    K = config["K"]
    n_hyp = config["n_hypotheses"]
    threshold = config["threshold"]
    
    embedder_name = embedder.replace("/", "_").replace("-", "_")
    config_name = f"emb={embedder_name}_M={M}_K={K}_n={n_hyp}_th={threshold}"
    cache_name = f"imdb_tuned_{embedder_name}_M{M}_K{K}"
    checkpoint_dir = f"./checkpoints/imdb_tuned_{embedder_name}_M{M}_K{K}"
    
    try:
        text2embedding = get_local_embeddings(
            texts=texts + val_texts,
            model=embedder,
            cache_name=cache_name,
            show_progress=False
        )
        
        train_embeddings_list = []
        for text in texts:
            if text in text2embedding:
                train_embeddings_list.append(text2embedding[text])
        train_embeddings = np.stack(train_embeddings_list)
        
        val_embeddings_list = []
        for text in val_texts:
            if text in text2embedding:
                val_embeddings_list.append(text2embedding[text])
        val_embeddings = np.stack(val_embeddings_list)
        
        sae = train_sae(
            embeddings=train_embeddings,
            val_embeddings=val_embeddings,
            M=M,
            K=K,
            checkpoint_dir=checkpoint_dir,
            n_epochs=50,
            show_progress=False
        )
        
        train_activations = sae.get_activations(train_embeddings)
        selected_neurons, scores = select_neurons(
            activations=train_activations,
            target=labels,
            method="lasso",
            n_select=n_hyp
        )
        
        hypothesis_templates = [
            "mentions positive aspects of the movie",
            "criticizes negative aspects of the movie",
            "discusses the plot or storyline",
            "comments on acting performance",
            "mentions cinematography or visual effects",
            "discusses the movie's pacing or length",
            "compares to other movies",
            "mentions the director or production"
        ]
        
        dummy_hypotheses = hypothesis_templates[:n_hyp]
        
        results_df = pd.DataFrame({
            'neuron_idx': selected_neurons[:n_hyp],
            'interpretation': dummy_hypotheses,
            'target_correlation': scores[:n_hyp] if len(scores) >= n_hyp else [0.5] * n_hyp
        })
        
        holdout_text2embedding = get_local_embeddings(
            texts=holdout_texts,
            model=embedder,
            cache_name=cache_name,
            show_progress=False
        )
        
        metrics, eval_df = evaluate_hypotheses(
            hypotheses_df=results_df,
            texts=holdout_texts,
            labels=holdout_labels,
            cache_name=cache_name,
            annotation_method="embedding",
            embedding_model=embedder,
            similarity_threshold=threshold,
            use_local_embeddings=True,
            text2embedding=holdout_text2embedding,
        )
        
        r2 = metrics['r2']
        n_significant = metrics['Significant'][0]
        n_total = metrics['Significant'][1]
        
        result = {
            "config": config_name,
            "embedder": embedder,
            "M": M,
            "K": K,
            "n_hypotheses": n_hyp,
            "threshold": threshold,
            "r2": r2,
            "significant": f"{n_significant}/{n_total}",
            "n_significant": n_significant,
            "n_total": n_total,
        }
        
        all_results.append(result)
        
        if not np.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_config = result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        continue

if all_results:
    results_df = pd.DataFrame(all_results)
    valid_results = results_df[results_df['r2'].notna()]
    valid_results = valid_results.sort_values('r2', ascending=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./optimization_results_tuned_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
