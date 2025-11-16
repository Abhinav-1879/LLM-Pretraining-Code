from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data.txt")
parser.add_argument("--out", default="dataset")
args = parser.parse_args()

with open(args.data, "r", encoding="utf-8") as f:
    text = f.read()

data = Dataset.from_dict({"text": text.split("\n")})
data = data.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def tok(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

ds = data.map(tok, batched=True, remove_columns=["text"])
ds.save_to_disk(args.out)
print("Saved:", args.out)
