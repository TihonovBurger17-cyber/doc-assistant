import pandas as pd


# === ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ (TOOLS) ===

def tool_database_search(query: str) -> str:
    """Инструмент 1: Поиск аниме по названию или жанру в датасете."""
    try:
        df = pd.read_csv("data/processed/clean.csv")

        # Ищем совпадения в колонке name или genre
        mask = (
                df['name'].astype(str).str.contains(query, case=False, na=False) |
                df['genre'].astype(str).str.contains(query, case=False, na=False)
        )
        results = df[mask].head(3)

        if not results.empty:
            found_list = []
            for _, row in results.iterrows():
                found_list.append(f"📌 Название: {row['name']} | Жанры: {row['genre']} | Тип: {row['type']}")
            return "Найдено в базе:\n" + "\n".join(found_list)
        else:
            # Если ничего не нашли, покажем первые парочку аниме из таблицы для примера
            fallback = df['name'].dropna().head(2).tolist()
            return f"По запросу '{query}' ничего не найдено. Попробуй поискать: 'gintama', 'steins;gate', 'haikyuu' или жанр 'Comedy'. Примеры в базе: {'; '.join(fallback)}"
    except Exception as e:
        return f"Ошибка при обращении к базе данных: {e}"


def tool_calculator(expression: str) -> str:
    """Инструмент 2: Выполнение математических расчетов."""
    try:
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

    def run(self, user_input: str) -> str:
        """Агент анализирует запрос и выбирает подходящий инструмент."""
        user_input_lower = user_input.lower()

        # Простая эвристика агента для выбора инструмента
        if any(op in user_input for op in ["+", "-", "*", "/", "посчитай", "вычисли"]):
            expr = "".join([c for c in user_input if c in "0123456789+-*/(). "])
            print("🤖 [Агент решил использовать инструмент]: calc")
            return self.tools["calc"](expr)
        else:
            print("🤖 [Агент решил использовать инструмент]: search")
            return self.tools["search"](user_input)


# === ТЕСТИРОВАНИЕ АГЕНТА ===
if __name__ == "__main__":
    agent = SimpleAgent()

    print("--- Тест 1 (Инструмент поиска по датасету аниме) ---")
    print(agent.run("steins"))

    print("\n--- Тест 2 (Инструмент калькулятора) ---")
    print(agent.run("посчитай 150 + 250 * 2"))