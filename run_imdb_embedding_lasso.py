"""Full dataset HypotheSAEs on IMDB using embedding similarity annotation and LASSO selection."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
# Set your OpenAI API key: os.environ['OPENAI_KEY_SAE'] = "your-api-key-here"

from hypothesaes.quickstart import train_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_openai_embeddings

current_dir = os.getcwd()
prefix = "./" if not current_dir.endswith("notebooks") else "../"
base_dir = os.path.join(prefix, "demo_data")

train_df = pd.read_json(os.path.join(base_dir, "imdb-demo-train-20K.json"), lines=True)
val_df = pd.read_json(os.path.join(base_dir, "imdb-demo-val-2K.json"), lines=True)
holdout_df = pd.read_json(os.path.join(base_dir, "imdb-demo-holdout-2K.json"), lines=True)

texts = train_df['text'].tolist()
labels = train_df['stars'].values
val_texts = val_df['text'].tolist()
holdout_texts = holdout_df['text'].tolist()
holdout_labels = holdout_df['stars'].values

EMBEDDER = "text-embedding-3-small"
CACHE_NAME = f"imdb_full_embedding_lasso_{EMBEDDER}"
text2embedding = get_openai_embeddings(
    texts + val_texts,
    model=EMBEDDER,
    cache_name=CACHE_NAME,
    n_workers=1
)

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

texts = texts_filtered
val_texts = val_texts_filtered
labels = np.array(labels_filtered)

train_embeddings = np.stack(train_embeddings_list)
val_embeddings = np.stack(val_embeddings_list)

checkpoint_dir = os.path.join(prefix, "checkpoints", CACHE_NAME)
M, K = 256, 8
sae = train_sae(
    embeddings=train_embeddings,
    val_embeddings=val_embeddings,
    M=M,
    K=K,
    checkpoint_dir=checkpoint_dir,
    n_epochs=100,
    batch_size=512,
    show_progress=True
)

TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are movie reviews from IMDB.
Features should describe a specific aspect of the review. For example:
- "mentions the director or specific filmmaking techniques"
- "criticizes the plot or storyline as confusing or poorly written"
- "praises the acting performance of specific actors"
"""

results = generate_hypotheses(
    texts=texts,
    labels=labels,
    embeddings=train_embeddings,
    sae=sae,
    cache_name=CACHE_NAME,
    selection_method="lasso",
    n_selected_neurons=20,
    n_candidate_interpretations=1,
    n_scoring_examples=0,
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    interpreter_model="gpt-4.1-mini",
    annotator_model="gpt-4.1-mini",
    n_workers_annotation=10,
    n_examples_for_interpretation=20,
    max_words_per_example=256,
)

holdout_text2embedding = get_openai_embeddings(
    holdout_texts,
    model=EMBEDDER,
    cache_name=CACHE_NAME,
    n_workers=1
)

metrics, evaluation_df = evaluate_hypotheses(
    hypotheses_df=results,
    texts=holdout_texts,
    labels=holdout_labels,
    cache_name=CACHE_NAME,
    annotation_method="embedding",
    embedding_model=EMBEDDER,
    similarity_threshold=0.7,
    use_local_embeddings=False,
    text2embedding=holdout_text2embedding,
)
