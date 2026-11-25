import re
import pandas as pd


def _format_row(row: pd.Series) -> str:
    parts = []
    for col in ["name", "genre", "type", "episodes", "rating", "members", "anime_id"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}={row[col]}")
    return ", ".join(parts) if parts else "запись без полей"


# === 1. ИНСТРУМЕНТЫ (3 ФУНКЦИИ ПО ВСЕМ КОЛОНКАМ CSV) ===

def tool_search_csv(query: str) -> str:
    """Поиск записей в data/processed/clean.csv по всем доступным колонкам."""
    df = pd.read_csv("data/processed/clean.csv")
    q = str(query).strip()
    if not q:
        return "Пустой поисковый запрос."

    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask = mask | df[col].astype(str).str.contains(q, case=False, na=False, regex=False)

    matches = df[mask].head(5)
    if matches.empty:
        return f"Не найдено по запросу: {q}"

    lines = [_format_row(row) for _, row in matches.iterrows()]
    return "Найдено: " + " ; ".join(lines)


def tool_count_csv(query: str = "") -> str:
    """Подсчёт всех строк в clean.csv или совпадений по любой колонке."""
    df = pd.read_csv("data/processed/clean.csv")
    q = str(query).strip()
    if not q or q.lower() in {"все", "всего", "строк", "записей", "count"}:
        return f"Всего записей в clean.csv: {len(df)}"

    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask = mask | df[col].astype(str).str.contains(q, case=False, na=False, regex=False)

    return f"Найдено строк по '{q}': {int(mask.sum())} из {len(df)}"


def tool_calc(expression: str) -> str:
    """Вычисление простого арифметического выражения (+, -, *, /, скобки)."""
    expr = str(expression).strip()
    if not expr or not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return f"Ошибка вычисления: некорректное выражение '{expr}'"
    result = eval(expr, {"__builtins__": {}})
    return f"{expr} = {result}"


TOOLS = {
    "search": tool_search_csv,
    "count": tool_count_csv,
    "calc": tool_calc,
}


# === 2. БЕЗОПАСНЫЙ ВЫЗОВ ИНСТРУМЕНТА ===

def run_tool(name: str, tool_input: str) -> str:
    """Вызывает инструмент из TOOLS и возвращает текст ошибки вместо падения."""
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS.keys())}"
    try:
        return str(TOOLS[name](tool_input))
    except Exception as e:
        return f"Ошибка инструмента '{name}': {e}"


# === 3. ЦИКЛ АГЕНТА С ЛОГИРОВАНИЕМ ШАГОВ ===

def run_agent_with_trace(user_query: str, max_steps: int = 5) -> dict:
    """
    Запускает цикл агента (до max_steps шагов), пишет лог каждого шага
    и формирует итоговый ответ для чат-бота.
    """
    trace = []
    observations = []

    parts = [p.strip() for p in re.split(r"\s+и\s+|,|;", user_query) if p.strip()]
    if not parts:
        parts = [user_query.strip()]

    for step in range(1, max_steps + 1):
        if step - 1 >= len(parts):
            break

        sub = parts[step - 1]
        sub_lower = sub.lower()

        math_match = re.search(r"([0-9]+(?:\s*[+\-*/]\s*[0-9]+)+)", sub)
        if math_match or any(w in sub_lower for w in ["посчитай", "вычисли", "сколько будет"]):
            action = "calc"
            tool_in = math_match.group(1) if math_match else sub
        elif any(w in sub_lower for w in ["сколько", "количество", "строк", "записей", "жанр", "тип"]):
            action = "count"
            cleaned = re.sub(
                r"(сколько|аниме|записей|строк|количество|посчитай|в жанре|по жанру|жанр|тип|в)",
                "",
                sub_lower,
            ).strip()
            tool_in = cleaned or "все"
        else:
            action = "search"
            cleaned = re.sub(r"(найди|найти|покажи|поиск|в файле|в базе|аниме)", "", sub_lower).strip()
            tool_in = cleaned or sub

        result = run_tool(action, tool_in)
        step_log = f"[шаг {step}] action={action} | input='{tool_in}' -> {result}"
        trace.append(step_log)
        observations.append(result)

    if not observations:
        reply = "Не нашёл ничего по запросу."
    else:
        has_not_found = any(obs.startswith("Не найдено") for obs in observations)
        has_found = any(
            obs.startswith("Найдено:")
            or obs.startswith("Всего записей")
            or obs.startswith("Найдено строк")
            or "=" in obs
            for obs in observations
        )
        if has_not_found and not has_found:
            reply = "Нет, не нашёл. " + " | ".join(observations)
        else:
            prefix = "Да, нашёл: " if any(obs.startswith("Найдено:") for obs in observations) else ""
            reply = prefix + " | ".join(observations)

    return {
        "reply": reply,
        "trace": trace,
    }


class SimpleAgent:
    """Совместимость со старым вызовом agent.run(text)."""

    def __init__(self):
        self.tools = TOOLS

    def run(self, user_input: str) -> str:
        return run_agent_with_trace(user_input)["reply"]