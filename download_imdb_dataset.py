"""Download IMDB dataset and convert to Yelp-like format."""

import json
import os
import random
from datasets import load_dataset

random.seed(42)

def convert_label_to_rating(label):
    if label == 0:
        return random.choice([1, 2])
    else:
        return random.choice([4, 5])

os.makedirs('demo_data', exist_ok=True)

train_ds = load_dataset('imdb', split='train[:20000]')
val_ds = load_dataset('imdb', split='test[:2000]')
holdout_ds = load_dataset('imdb', split='test[2000:4000]')

train_data = [{'text': item['text'], 'stars': convert_label_to_rating(item['label'])} for item in train_ds]
with open('demo_data/imdb-demo-train-20K.json', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

val_data = [{'text': item['text'], 'stars': convert_label_to_rating(item['label'])} for item in val_ds]
with open('demo_data/imdb-demo-val-2K.json', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

holdout_data = [{'text': item['text'], 'stars': convert_label_to_rating(item['label'])} for item in holdout_ds]
with open('demo_data/imdb-demo-holdout-2K.json', 'w', encoding='utf-8') as f:
    for item in holdout_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
