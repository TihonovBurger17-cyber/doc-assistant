import os
import openai
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================================================================
# РАЗДЕЛ 1: Функция для работы с API LLM
# =====================================================================
def call_llm_api(
    prompt: str,
    system_prompt: str = "Ты — полезный ассистент.",
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 500,
    json_mode: bool = False,
    api_key: str = None,
) -> str:
    """Универсальная функция вызова LLM API (OpenAI API / vLLM / LocalAI)

    Параметры:
      - temperature: 0.0-0.2 для строгого JSON/анализа, 0.7+ для творчества
      - json_mode: Если True, гарантирует ответ строго в формате JSON
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "ВАШ_API_КЛЮЧ")
    client = openai.OpenAI(api_key=key)

    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# =====================================================================
# РАЗДЕЛ 2: Настройка локальной модели с LoRA (PEFT)
# =====================================================================
def setup_lora_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
):
    """Загружает локальную базовую модель и оборачивает её в LoRA-адаптер."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    print(f"Загрузка базовой модели: {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    print("\n--- Информация о параметрах LoRA ---")
    model.print_trainable_parameters()

    return model


if __name__ == "__main__":
    # --- ТЕСТ 1: Вызов API (требуется рабочий API ключ) ---
    #Если нет API ключа OpenAI, закомментируй этот блок
    try:
        print("=== Тестирование вызова API ===")
        res = call_llm_api(
            prompt="Проанализируй текст и выдели три главных ключевых слова: 'Нейросети прочно вошли в нашу жизнь, автоматизируя написание кода и генерацию контента.'",
            system_prompt="Ты — бот-аналитик. Отвечай строго в формате JSON.",
            temperature=0.2,
            json_mode=True,
            api_key="ТВОЙ_OPENAI_API_KEY" # Замени на реальный ключ или задай через os.environ["OPENAI_API_KEY"]
        )
        print("Ответ API:")
        print(res)
    except Exception as e:
        print(f"Ошибка вызова API (проверь API ключ): {e}")

    print("\n" + "="*40 + "\n")

    # --- ТЕСТ 2: Инициализация локальной LLM с LoRA ---
    print("=== Инициализация LoRA модели ===")
    # Используем компактную модель для локального теста
    model = setup_lora_model(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", # Легкая версия Qwen для быстрой проверки
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )