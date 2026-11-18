import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai


def build_index():
    """Строит или загружает векторный индекс один раз при старте[cite: 31]."""
    df = pd.read_csv("data/processed/clean.csv")
    text_col = df.select_dtypes(include=["object", "string"]).columns[0]
    texts = df[text_col].dropna().tolist()[:100]  # Первые 100 записей

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    return vector_db


def rag_answer(query: str, vector_db, k: int = 3):
    """Выполняет поиск чанков и генерирует ответ через LLM[cite: 27, 31]."""
    if vector_db is None:
        vector_db = build_index()

    top_k_docs = vector_db.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in top_k_docs])
    sources = [f"chunk_{i}" for i in range(len(top_k_docs))]

    prompt = (
        "Ответь на вопрос ТОЛЬКО по приведённым фрагментам. "
        "Если ответа в них нет, скажи 'в документах не найдено'.\n\n"
        f"Фрагменты:\n{context}\n\n"
        f"Вопрос: {query}"
    )

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "ВАШ_API_КЛЮЧ")
    if api_key and api_key != "ВАШ_API_КЛЮЧ":
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content
    else:
        answer = f"[Mock Ответ по контексту]: Найдено чанков: {len(top_k_docs)} для запроса '{query}'."

    return answer, sources