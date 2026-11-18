import pandas as pd


# === ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ (TOOLS) ===

def tool_database_search(query: str) -> str:
    """Инструмент 1: Поиск информации в нашем датасете."""
    try:
        df = pd.read_csv("data/processed/clean.csv")
        text_col = df.select_dtypes(include=["object", "string"]).columns[0]
        results = df[df[text_col].str.contains(query, case=False, na=False)][text_col].head(2).tolist()
        if results:
            return f"Найдено в базе: {'; '.join(results)}"
        return "В базе данных ничего не найдено по вашему запросу."
    except Exception as e:
        return f"Ошибка при обращении к базе данных: {e}"


def tool_calculator(expression: str) -> str:
    """Инструмент 2: Выполнение математических расчетов."""
    try:
        # Безопасное вычисление простых математических выражений
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Ошибка: недопустимые символы в математическом выражении."
        result = eval(expression)
        return f"Результат расчета: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {e}"


# === ЛОГИКА АГЕНТА ===

class SimpleAgent:
    def __init__(self):
        self.tools = {
            "search": tool_database_search,
            "calc": tool_calculator
        }

    pdef
    run(self, user_input: str) -> str:
    """Агент анализирует запрос и выбирает подходящий инструмент."""
    user_input_lower = user_input.lower()

    # Простая эвристика агента для выбора инструмента
    if any(op in user_input for op in ["+", "-", "*", "/", "посчитай", "вычисли"]):
        # Выделяем выражение для калькулятора
        expr = "".join([c for c in user_input if c in "0123456789+-*/(). "])
        print("🤖 [Агент решил использовать инструмент]: calc")
        return self.tools["calc"](expr)
    else:
        print("🤖 [Агент решил использовать инструмент]: search")
        return self.tools["search"](user_input)


# === ТЕСТИРОВАНИЕ АГЕНТА ===
if __name__ == "__main__":
    agent = SimpleAgent()

    print("--- Тест 1 (Инструмент поиска) ---")
    print(agent.run("аниме"))

    print("\n--- Тест 2 (Инструмент калькулятора) ---")
    print(agent.run("посчитай 150 + 250 * 2"))