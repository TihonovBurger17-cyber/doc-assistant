import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
df = pd.read_csv("data/processed/clean.csv")

text_col = df.select_dtypes(include=["object", "string"]).columns[0]

# Ищем подходящую категориальную колонку для классификации (type или genre)
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
    # Если категорий нет, создаем синтетические бинарные классы по длине названия/описания
    df['label'] = (df[text_col].str.len() > df[text_col].str.len().median()).astype(int)

# Увеличиваем словарь TF-IDF до 1500 признаков и добавляем биграммы (ngram_range)
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), stop_words='english')
X_vec = vectorizer.fit_transform(df[text_col]).toarray()
y_vec = df['label'].values

num_classes = len(set(y_vec))
in_features = X_vec.shape[1]

# Разделение 80/20 по слайду
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_vec, test_size=0.2, random_state=42, stratify=y_vec
)

# Преобразование в тензоры PyTorch
X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test, dtype=torch.float32)
y_te = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# -------------------------------------------------------------
# 2. Определение 3-х архитектур моделей
# -------------------------------------------------------------

class Model1_Base(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class Model2_Deep(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class Model3_Dropout(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


models = {
    "Model 1 (Base)": Model1_Base(in_features, num_classes),
    "Model 2 (Deep + BatchNorm)": Model2_Deep(in_features, num_classes),
    "Model 3 (Dropout)": Model3_Dropout(in_features, num_classes)
}

# -------------------------------------------------------------
# 3. Обучение (30 эпох) и вычисление Accuracy
# -------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()

for name, model in models.items():
    print(f"\n=================== Обучение {name} ===================")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 31):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Оценка на тесте (согласно правилам на слайде)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te)
        preds = torch.argmax(test_logits, dim=1)
        acc = (preds == y_te).float().mean().item()
        print(f"Результат {name} — Accuracy на тесте: {acc * 100:.2f}%")