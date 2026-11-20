from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.rag import build_index, rag_answer
from src.agent import run_agent_with_trace

app = FastAPI(title="Doc Assistant")

index_state = None


@app.on_event("startup")
def load():
    global index_state
    print("🚀 Инициализация и построение RAG индекса...")
    index_state = build_index()
    print("✅ RAG индекс успешно загружен в память!")


@app.get("/health")
def health():
    return {"status": "ok"}


class Query(BaseModel):
    question: str
    k: int = 3


@app.post("/ask")
def ask(q: Query):
    if not q.question or not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    answer, sources = rag_answer(q.question, index_state, k=q.k)
    return {"answer": answer, "sources": sources}


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустое сообщение")

    result = run_agent_with_trace(text)
    return {
        "reply": result["reply"],
        "trace": result["trace"],
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <title>Чат-бот Агент по файлу</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 760px; margin: 32px auto; background: #111; color: #eee; }
            #chat { border: 1px solid #333; background: #1a1a1a; padding: 14px; height: 340px; overflow-y: auto; margin-bottom: 12px; white-space: pre-wrap; }
            #trace { border: 1px solid #2b2b2b; background: #151515; color: #aaa; padding: 10px; min-height: 80px; font-size: 13px; margin-bottom: 12px; white-space: pre-wrap; }
            .u { color: #7cc4fa; margin: 6px 0; }
            .b { color: #9be79b; margin: 6px 0; }
            form { display: flex; gap: 8px; }
            input { flex: 1; padding: 10px; background: #242424; color: #fff; border: 1px solid #444; }
            button { padding: 10px 18px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h2>Чат-бот агент (поиск по файлу + инструменты)</h2>
        <div id="chat"></div>
        <div><strong>Лог шагов агента:</strong></div>
        <div id="trace"></div>
        <form id="form">
            <input id="msg" type="text" placeholder="Например: steins, Comedy, сколько строк, посчитай 15+25" autocomplete="off" />
            <button type="submit">Отправить</button>
        </form>
        <script>
            const chat = document.getElementById("chat");
            const traceBox = document.getElementById("trace");
            const msgInput = document.getElementById("msg");

            function appendLine(cls, text) {
                const el = document.createElement("div");
                el.className = cls;
                el.textContent = text;
                chat.appendChild(el);
                chat.scrollTop = chat.scrollHeight;
            }

            document.getElementById("form").addEventListener("submit", async (e) => {
                e.preventDefault();
                const text = msgInput.value.trim();
                if (!text) return;
                appendLine("u", "Вы: " + text);
                msgInput.value = "";

                const res = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                appendLine("b", "Агент: " + data.reply);
                traceBox.textContent = (data.trace || []).join("\\n");
            });
        </script>
    </body>
    </html>
    """