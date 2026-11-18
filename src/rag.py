import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai

# 1. Загрузка очищенного датасета
df = pd.read_csv("data/processed/clean.csv")
text_col = df.select_dtypes(include=["object", "string"]).columns[0]
texts = df[text_col].dropna().tolist()[:100]  # Первые 100 записей

# 2. Нарезка на чанки (500 символов, перекрытие 50)[cite: 26]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.create_documents(texts)
print(f"✅ Создано чанков: {len(docs)}")

# 3. Эмбеддинги (BERT) и векторная БД (Chroma)[cite: 24, 26]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# 4. Поиск Top-K чанков по запросу (ANN-поиск)[cite: 27]
query = "Расскажи про интересные аниме"
top_k_docs = vector_db.similarity_search(query, k=3)

print(f"\n🔍 Вопрос пользователя: {query}")
print("--- Найденные Top-K чанки (Контекст) ---")
for i, doc in enumerate(top_k_docs, 1):
    print(f"\n[Чанк {i}]:\n{doc.page_content[:200]}...")

# 5. Сборка промпта и вызов LLM (с защитой от галлюцинаций)[cite: 27]
context = "\n\n".join([d.page_content for d in top_k_docs])

prompt = (
    "Ответь на вопрос ТОЛЬКО по приведённым фрагментам. "
    "Если ответа в них нет, скажи 'в документах не найдено'.\n\n"
    f"Фрагменты:\n{context}\n\n"
    f"Вопрос: {query}"
)

print("\n--- Отправка запроса в LLM ---")
# Замени на свой реальный API ключ, если нужно протестировать генерацию
api_key = os.getenv("OPENAI_API_KEY", "ВАШ_API_КЛЮЧ")

if api_key != "ВАШ_API_КЛЮЧ":
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    print("\n🤖 Итоговый ответ LLM:")
    print(response.choices[0].message.content)
else:
    print("⚠️ API ключ не задан. Сгенерированный промпт для LLM:")
    print("-" * 40)
    print(prompt[:500] + "...\n[усечено]")