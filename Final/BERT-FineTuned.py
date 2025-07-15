import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

try:
    dataset = load_dataset("imdb")
except Exception as e:
    print(f"{e}")
    


train_dataset = dataset["train"].shuffle(seed=42).select(range(2500))
test_dataset = dataset["test"].shuffle(seed=42).select(range(500)) 

print("Sampled")
print(train_dataset)
print(test_dataset)
