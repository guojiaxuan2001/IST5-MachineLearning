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
    print(f"加载数据集失败: {e}")
    print("请尝试重置 Colab 运行环境后再试！")

# 为了快速演示，我们只取一小部分数据
# 如果你的时间和GPU允许，可以增加样本量
train_dataset = dataset["train"].shuffle(seed=42).select(range(2500)) # 训练集增加到2500
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))  # 测试集增加到500

print("数据集加载成功并已抽样：")
print(train_dataset)
print(test_dataset)