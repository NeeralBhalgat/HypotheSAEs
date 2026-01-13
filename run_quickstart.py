"""Run the quickstart notebook cells sequentially."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
# Set your OpenAI API key: os.environ['OPENAI_KEY_SAE'] = "your-api-key-here"

from hypothesaes.quickstart import train_sae, interpret_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_openai_embeddings

INTERPRETER_MODEL = "gpt-4.1"
ANNOTATOR_MODEL = "gpt-4.1-mini"
N_WORKERS_ANNOTATION = 10

current_dir = os.getcwd()
prefix = "../" if current_dir.endswith("notebooks") else "./"
base_dir = os.path.join(prefix, "demo_data")

train_df = pd.read_json(os.path.join(base_dir, "yelp-demo-train-20K.json"), lines=True)
val_df = pd.read_json(os.path.join(base_dir, "yelp-demo-val-2K.json"), lines=True)
texts = train_df['text'].tolist()
labels = train_df['stars'].values
val_texts = val_df['text'].tolist()

EMBEDDER = "text-embedding-3-small"
CACHE_NAME = f"yelp_quickstart_{EMBEDDER}"
text2embedding = get_openai_embeddings(texts + val_texts, model=EMBEDDER, cache_name=CACHE_NAME)
embeddings = np.stack([text2embedding[text] for text in texts])
train_embeddings = np.stack([text2embedding[text] for text in texts])
val_embeddings = np.stack([text2embedding[text] for text in val_texts])

checkpoint_dir = os.path.join(prefix, "checkpoints", CACHE_NAME)
sae = train_sae(embeddings=train_embeddings, val_embeddings=val_embeddings,
                M=256, K=8, matryoshka_prefix_lengths=[32, 256], 
                checkpoint_dir=checkpoint_dir)

TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are reviews of restaurants on Yelp.
Features should describe a specific aspect of the review. For example:
- "mentions long wait times to receive service"
- "praises how a dish was cooked, with phrases like 'perfect medium-rare'"""

results = interpret_sae(
    texts=texts,
    embeddings=train_embeddings,
    sae=sae,
    n_random_neurons=5,
    print_examples_n=3,
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    interpreter_model=INTERPRETER_MODEL,
)

selection_method = "correlation"
results = generate_hypotheses(
    texts=texts,
    labels=labels,
    embeddings=embeddings,
    sae=sae,
    cache_name=CACHE_NAME,
    selection_method=selection_method,
    n_selected_neurons=20,
    n_candidate_interpretations=1,
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    interpreter_model=INTERPRETER_MODEL,
    annotator_model=ANNOTATOR_MODEL,
    n_workers_annotation=N_WORKERS_ANNOTATION,
)

holdout_df = pd.read_json(os.path.join(base_dir, "yelp-demo-holdout-2K.json"), lines=True)
holdout_texts = holdout_df['text'].tolist()
holdout_labels = holdout_df['stars'].values

metrics, evaluation_df = evaluate_hypotheses(
    hypotheses_df=results,
    texts=holdout_texts,
    labels=holdout_labels,
    cache_name=CACHE_NAME,
    annotator_model=ANNOTATOR_MODEL,
    n_workers_annotation=N_WORKERS_ANNOTATION,
)
