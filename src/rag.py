import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Загрузка очищенного датасета
df = pd.read_csv("data/processed/clean.csv")
text_col = df.select_dtypes(include=["object", "string"]).columns[0]
texts = df[text_col].dropna().tolist()[:100]  # Первые 100 записей

# 2. Нарезка на чанки (Параметры 500 / 50 строго по доске)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.create_documents(texts)
print(f"Успешно создано чанков: {len(docs)}")

# 3. Эмбеддинги (BERT) и Векторная БД (Chroma)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# 4. Поиск Top-K чанков по вопросу
query = "аниме про магию и приключения"
top_k_docs = vector_db.similarity_search(query, k=3)

print(f"\nВопрос: {query}")
print("--- Найденные Top-K чанки (Контекст) ---")
for i, doc in enumerate(top_k_docs, 1):
    print(f"\nЧанк {i}:\n{doc.page_content[:200]}...")

# 5. Сборка итогового промпта (top-k + вопрос)
context = "\n\n".join([d.page_content for d in top_k_docs])
final_prompt = f"Контекст из базы:\n{context}\n\nВопрос пользователя: {query}\nОтветь на вопрос только на основе контекста и укажи источники."

print("\n--- Финальный промпт для LLM API ---")
print(final_prompt[:500] + "...")