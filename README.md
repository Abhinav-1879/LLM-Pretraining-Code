# LLM Pretraining Code

A small end-to-end workflow demonstrating dataset preparation and fine-tuning a causal language model (GPT-2) using HuggingFace.

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare dataset from `data.txt`:
   python data_prep.py --data data.txt

3. Train the model:
   python train.py

## Files
- data_prep.py
- train.py
- requirements.txt
