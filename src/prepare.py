import os
import pandas as pd

#Поиск CSV файла в папке
data_dir = "data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("Положи свой CSV-файл в папку 'data/'!")

input_path = os.path.join(data_dir, csv_files[0])
print(f"Загрузка файла: {input_path}")
df = pd.read_csv(input_path)

#  Поиск текстовой колонки для фильтрации
text_cols = df.select_dtypes(include=["object"]).columns

if len(text_cols) > 0:
    text_col = text_cols[0]
    print(f"Обработка текстовой колонки: '{text_col}'")

    # Дубли, Пропуски, Длина < 50
    df = df.drop_duplicates()
    df = df.dropna(subset=[text_col])

    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.len() >= 50]

    # Приведение к нижнему регистру
    df[text_col] = df[text_col].str.lower()
else:
    # числовые ID и оценки
    df = df.drop_duplicates()
    df = df.dropna()

# Сохранение результатов в data/processed/clean.csv
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/clean.csv"
df.to_csv(output_path, index=False)

print(f"Успешно сохранено в {output_path}. Итого строк: {len(df)}")