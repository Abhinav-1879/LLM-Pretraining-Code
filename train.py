from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

ds = load_from_disk("dataset")
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tok))

if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir="model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=20,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

trainer.train()
trainer.save_model("model")
print("Model saved to model/")
