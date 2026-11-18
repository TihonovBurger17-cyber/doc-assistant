from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag import build_index, rag_answer

app = FastAPI(title="Doc Assistant")

# Глобальное состояние для индекса (грузим 1 раз при старте, а не на каждый запрос)[cite: 31]
index_state = None


@app.on_event("startup")
def load():
    global index_state
    print("🚀 Инициализация и построение RAG индекса...")
    index_state = build_index()
    print("✅ RAG индекс успешно загружен в память!")


@app.get("/health")
def health():
    """Эндпоинт проверки здоровья сервиса."""
    return {"status": "ok"}


class Query(BaseModel):
    question: str
    k: int = 3


@app.post("/ask")
def ask(q: Query):
    """Эндпоинт обработки вопросов через RAG с валидацией и обработкой ошибок[cite: 30, 31]."""
    if not q.question or not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    answer, sources = rag_answer(q.question, index_state, k=q.k)
    return {"answer": answer, "sources": sources}