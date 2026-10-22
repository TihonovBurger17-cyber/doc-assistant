import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# 1. Загрузка данных
df = pd.read_csv("data/processed/clean.csv")
text_col = df.select_dtypes(include=["object", "string"]).columns[0]

target_col = None
for col in ['type', 'genre', 'rating']:
    if col in df.columns:
        target_col = col
        break

if target_col:
    df = df.dropna(subset=[text_col, target_col])
    le = LabelEncoder()
    df['label'] = le.fit_transform(df[target_col].astype(str))
else:
    df['label'] = (df[text_col].str.len() > df[text_col].str.len().median()).astype(int)

num_labels = len(set(df['label']))

train_df, test_df = train_test_split(df[[text_col, 'label']], test_size=0.2, random_state=42)

# 2. Токенизация
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples[text_col], truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 3. Базовая модель
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 4. LoRA Config с исправленным target_modules
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# 5. Параметры обучения
training_args = TrainingArguments(
    output_dir="./results_lora",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("\n--- Запуск обучения с LoRA ---")
trainer.train()

eval_results = trainer.evaluate()
print(f"\nРезультаты LoRA — Eval Loss: {eval_results['eval_loss']:.4f}")