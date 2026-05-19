# План внедрения: Уточняющий диалог перед RAG-поиском

> Документ-источник: [dialog_feature_description.md](dialog_feature_description.md)
>
> **Стратегия:** поэтапная (MVP = фазы 1–3, фаза 4 отдельно).
> **FSM storage:** `MemoryStorage` (как в [bot/auth/handler.py](bot/auth/handler.py)).
> **UI уточнений:** inline-кнопки + `/skip` + свободный текст.
> **Логирование:** только `clarification_rounds` в `query_logs` (минималистично).

---

## 0. Архитектурные решения до старта

### Что меняется в pipeline

Текущий путь — [bot/handlers/user.py:280-342](bot/handlers/user.py#L280-L342):
```
text → handle_question → retriever.search(question) → generator.generate
```

Новый путь:
```
text → classify_intent → [быстрый путь]  → retriever.search(question)
                       ↓ [расплывчатый]
                       clarification_loop (FSM, 1-3 раунда)
                       ↓
                       enrich_query
                       ↓
                       retriever.search(enriched_query)
                       ↓
                       generator.generate
```

### Что НЕ меняется на MVP

- `Retriever.search()` остаётся как есть — обогащённый запрос просто длиннее.
- `Generator.generate()` не трогается.
- FAQ / админка / feedback / verification — без изменений.

### Новый модуль

Создаётся новый пакет `rag/dialog/` рядом с `rag/retriever.py`:

```
rag/dialog/
├── __init__.py
├── classifier.py     # Фаза 1: классификация намерения
├── question_gen.py   # Фаза 2: генерация уточняющих вопросов
├── enricher.py       # Фаза 3: обогащение запроса
└── prompts.py        # Все LLM-промпты в одном месте
```

Аргументация: вся LLM-логика диалога — отдельная подсистема, не связана с retrieval/generation в их текущем виде. Изоляция упростит будущую замену моделей и тестирование.

---

## ЭТАП 1 — Фаза 1: Классификация намерения

**Цель:** определить, нужны ли уточнения. Если нет — поведение остаётся прежним.

### 1.1. `rag/dialog/classifier.py`

Функция `classify_intent(question: str) -> ClassificationResult`:

```python
class ClassificationResult(TypedDict):
    needs_clarification: bool
    reason: str           # для логов: "specific" / "vague_topic" / "ambiguous"
    suggested_topics: list[str]   # подсказки для question_gen, опционально
```

Реализация — один LLM-вызов через `_make_llm_client()` из [rag/generator.py:124-133](rag/generator.py#L124-L133) с low-temperature системным промптом, возвращающим строгий JSON. На ошибку парсинга — fallback `needs_clarification=False` (деградируем в текущее поведение).

### 1.2. Промпт классификатора (`rag/dialog/prompts.py`)

Промпт включает:
- список типичных «конкретных» vs «расплывчатых» формулировок (примеры из описания);
- инструкцию вернуть только JSON;
- многоязычность (RU/EN/KZ) — классификатор должен работать на языке вопроса.

### 1.3. Интеграция в `handle_question`

В [bot/handlers/user.py:281](bot/handlers/user.py#L281) после rate-limit проверки:

```python
intent = await classify_intent(question)
if not intent["needs_clarification"]:
    # текущая логика без изменений
    chunks = await _retriever.search(question)
    ...
else:
    # см. ЭТАП 2
    await start_clarification_dialog(message, question, state)
    return
```

### 1.4. Тестирование Этапа 1

- Прогнать через бота ~10 «конкретных» и ~10 «расплывчатых» запросов вручную.
- Принять решение: если классификатор ошибается >30% — пересмотреть промпт, при <15% — переходить к Этапу 2.

**Артефакты этапа:** `rag/dialog/classifier.py`, `rag/dialog/prompts.py`, изменения в [bot/handlers/user.py](bot/handlers/user.py).

---

## ЭТАП 2 — Фаза 2: Уточняющий диалог (FSM)

### 2.1. Подключение FSM-инфраструктуры

В [bot/main.py](bot/main.py) убедиться, что `Dispatcher` создаётся с `storage=MemoryStorage()` (как для auth). Если auth уже использует общий storage — переиспользуем его.

### 2.2. Состояния FSM — `bot/handlers/dialog_states.py`

```python
class ClarifyDialog(StatesGroup):
    waiting_for_answer = State()
```

В `state.set_data()` сохраняем:
```python
{
    "original_query": str,
    "rounds_done": int,            # 0..3
    "answers": list[dict],         # [{"question": str, "answer": str}]
    "status_msg_id": int,          # чтобы редактировать сообщение, а не плодить
}
```

### 2.3. Генератор уточняющих вопросов — `rag/dialog/question_gen.py`

Функция `next_clarification(state_data, available_docs) -> ClarificationQuestion`:

```python
class ClarificationQuestion(TypedDict):
    question: str           # текст вопроса на языке запроса
    options: list[str]      # 2-4 варианта для inline-кнопок (может быть [])
    stop: bool              # LLM решил, что достаточно — не задавать вопрос
```

Входы для LLM:
- `original_query` пользователя;
- список доступных документов (`doc_title`, `section_title`) — берём из `Retriever.get_all_documents()` ([rag/retriever.py:178-213](rag/retriever.py#L178-L213)); кешируется in-memory на старте + TTL ~10 минут;
- история уже заданных вопросов и ответов;
- номер раунда.

### 2.4. Хендлеры диалога — `bot/handlers/dialog.py`

Новый router, регистрируется в [bot/main.py](bot/main.py) **до** существующих text-handlers, чтобы перехватывать ответы в состоянии `ClarifyDialog.waiting_for_answer`.

Три хендлера:

1. **`@router.callback_query(F.data.startswith("clarify:"))`** — пользователь нажал inline-кнопку (`clarify:<round>:<option_index>`). Сохраняем ответ в `state`, идём в следующий раунд или к обогащению.

2. **`@router.message(Command("skip"), ClarifyDialog.waiting_for_answer)`** — пользователь пропустил уточнение. Помечаем текущий раунд как пропущенный, либо завершаем диалог если решили хватит.

3. **`@router.message(F.text, ClarifyDialog.waiting_for_answer)`** — свободный текст. Триггеры стопа: тексты «не знаю» / «неважно» / «не важно» / «idk» / «whatever» / «pass» (case-insensitive) — завершаем диалог. Иначе — записываем как ответ.

### 2.5. Цикл уточнений — `start_clarification_dialog` и `_ask_next`

```python
async def start_clarification_dialog(message, question, state):
    available_docs = await _get_cached_docs()
    next_q = await next_clarification(
        state_data={"original_query": question, "rounds_done": 0, "answers": []},
        available_docs=available_docs,
    )
    if next_q["stop"]:        # LLM решил, что уточнения не нужны
        await _proceed_to_search(message, question, state, answers=[])
        return
    await state.set_state(ClarifyDialog.waiting_for_answer)
    await state.set_data({...})
    await _send_clarification(message, next_q, round_no=0)

async def _ask_next(message, state):
    data = await state.get_data()
    if data["rounds_done"] >= 3:
        await _proceed_to_search(...)
        return
    next_q = await next_clarification(data, await _get_cached_docs())
    if next_q["stop"]:
        await _proceed_to_search(...)
        return
    await _send_clarification(message, next_q, round_no=data["rounds_done"])
```

### 2.6. Сборка inline-кнопок

```python
def clarify_keyboard(round_no: int, options: list[str]) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=opt, callback_data=f"clarify:{round_no}:{i}")]
        for i, opt in enumerate(options[:4])
    ]
    buttons.append([InlineKeyboardButton(text="⏭ Пропустить", callback_data=f"clarify:{round_no}:skip")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)
```

Telegram callback_data ограничен 64 байтами → передаём только индекс варианта; сам текст лежит в `state`.

### 2.7. Тестирование Этапа 2

- Сценарий: расплывчатый запрос → 1-3 кнопочных ответа → запрос отправляется в обычный поиск (на Этапе 3 заменим на обогащённый).
- Сценарий: `/skip` на первом вопросе → переход к поиску.
- Сценарий: «не знаю» свободным текстом → переход к поиску.
- Перезапустить бота во время диалога — пользователь должен корректно получить сообщение «начните заново» (либо безболезненно начать новый диалог следующим сообщением).

**Артефакты этапа:** `rag/dialog/question_gen.py`, `bot/handlers/dialog.py`, `bot/handlers/dialog_states.py`, регистрация router в [bot/main.py](bot/main.py).

---

## ЭТАП 3 — Фаза 3: Обогащение запроса

### 3.1. `rag/dialog/enricher.py`

Функция `enrich_query(original: str, answers: list[dict]) -> str`:

Один LLM-вызов: на вход — оригинальный запрос + список Q/A пар; на выход — единая поисковая строка (как в примере описания: «льготы на проживание в общежитии для студентов бакалавриата 2024-2025»).

Структурированный `user_profile` из описания (`topics` / `user_type` / `document_hints` / `temporal_context`) **на MVP не реализуем** — он нужен только для многофакторного скоринга (Фаза 4). Достаточно текстового обогащения, т.к. retriever — это cosine по embedding строки.

### 3.2. Интеграция в pipeline

В `_proceed_to_search` (создан на Этапе 2):

```python
async def _proceed_to_search(message, original_query, state, answers):
    if answers:
        enriched = await enrich_query(original_query, answers)
    else:
        enriched = original_query
    chunks = await _retriever.search(enriched)
    result = await _generator.generate(original_query, chunks)
    # NB: в generator передаём ОРИГИНАЛЬНЫЙ запрос, чтобы ответ не выглядел странно
    ...
    await state.clear()
```

Важный нюанс: в LLM-генератор уходит **оригинальный** вопрос пользователя (иначе ответ зазвучит как ответ на «расширенный машинный запрос»). Обогащение — только для retrieval.

### 3.3. Логирование

Расширить `log_query` в [bot/handlers/feedback.py:22-44](bot/handlers/feedback.py#L22-L44) — добавить опциональный параметр `clarification_rounds: int = 0`.

SQL-миграция (выполнится при старте, как в существующих модулях):
```sql
ALTER TABLE query_logs ADD COLUMN clarification_rounds INTEGER DEFAULT 0;
```

Миграцию положить в инициализатор БД (найти место, где создаётся `query_logs` — вероятно в [bot/main.py](bot/main.py) или в feedback). Использовать паттерн `try ALTER … except OperationalError: pass` для идемпотентности.

### 3.4. Тестирование Этапа 3

- Запрос «Какие есть льготы?» → 2 ответа → проверить в логах БД, что `clarification_rounds=2` и что ответ LLM содержит конкретику по уточнённой теме.
- A/B вручную: тот же расплывчатый запрос **без** уточнений (форсированно через debug) vs **с** уточнениями — сравнить релевантность найденных чанков.

**Артефакты этапа:** `rag/dialog/enricher.py`, миграция БД, обновление `log_query`.

---

## ЭТАП 4 — Фаза 4: Многофакторный скоринг + LLM-реранкинг (отдельный PR)

**Запускается только после стабилизации Этапов 1-3 в продакшне.**

### 4.1. Профиль пользователя

Расширить `enricher.py` — `extract_profile(original, answers) -> UserProfile`:
```python
class UserProfile(TypedDict):
    topics: list[str]
    user_type: Optional[str]       # "бакалавр" | "магистрант" | "сотрудник" | None
    document_hints: list[str]
    temporal_context: Optional[str]
```

Один LLM-вызов с инструкцией вернуть строгий JSON; на parse error — пустой профиль (graceful degradation в обычный поиск).

### 4.2. Многофакторный скоринг — `rag/retriever.py`

Новый метод `Retriever.search_with_profile(enriched_query, profile, k)`:
1. Поднять `limit=settings.top_k * 5` кандидатов из Qdrant.
2. Для каждого вычислить взвешенную сумму факторов (веса из описания):
   - 0.35 × cosine (есть из коробки);
   - 0.20 × match на `user_type` (substring / fuzzy в тексте чанка);
   - 0.15 × match на `document_hints` (по `doc_title` / `filename`);
   - 0.10 × positional bonus (по `paragraph_range` / признаку «заголовок»);
   - 0.10 × recency (по `uploaded_at` относительно сегодня);
   - 0.10 × cosine между `section_title` и `enriched_query` (нужен дополнительный embedder-вызов; кеш по section_title).
3. Сортировать по итоговому скору, вернуть top-10.

### 4.3. LLM-реранкер

Функция `rerank_chunks(query, chunks, k=5) -> list[dict]`:
- Один LLM-вызов: на вход — query и пронумерованные чанки (заголовок + первые ~300 символов текста);
- LLM возвращает JSON: список индексов в новом порядке + 1 строка обоснования каждый;
- Чанки переставляются согласно ответу; обоснования логируются (DEBUG, не пользователю).

### 4.4. Решение по логированию Фазы 4

`reranker_scores` / `user_profile` в `query_logs` — на этом этапе явно отказались. Если на Фазе 4 понадобится дебаг — добавить временный DEBUG-лог в файл (без поля в БД), и принять решение по схеме отдельно.

**Артефакты этапа:** изменения в `rag/retriever.py`, новый `rag/dialog/reranker.py`, расширение `enricher.py`.

---

## Риски и митигации

| Риск | Митигация |
|---|---|
| LLM-классификатор слишком часто триггерит уточнения | Метрика на Этапе 1 (ручное измерение на 20 запросах); порог «расплывчатости» ужесточается в промпте |
| Пользователь раздражается лишними вопросами | `/skip` всегда виден; LLM может остановить диалог досрочно (`stop: true`); жёсткий потолок 3 раунда |
| Дополнительные LLM-вызовы (classify + N×question_gen + enrich) → задержка | Все вызовы — с low max_tokens (50-200); классификатор температурой 0; на Этапе 1 измерить общую задержку на расплывчатом запросе — цель < 5 сек до первого вопроса |
| Гонка: пользователь шлёт новый текстовый вопрос, находясь в состоянии диалога | Хендлер `F.text, ClarifyDialog.waiting_for_answer` уже ловит — текущее сообщение трактуется как ответ на уточнение; для выхода — `/skip` |
| Перезагрузка бота во время диалога → MemoryStorage пуст | Принято осознанно: пользователь просто отправит вопрос заново. Альтернативу (SQLite-storage) отложили до появления реальной боли |
| Стоимость LLM-вызовов растёт | На Этапе 1 после прогона снять статистику: сколько % запросов идут «быстрым» путём без уточнений (цель: 60-70%) |

---

## Чеклист на выпуск каждого этапа

**Перед merge:**
- [ ] Ручное тестирование 10+ запросов разной сложности
- [ ] Проверка работы рейт-лимитера в новом потоке
- [ ] Проверка многоязычности (RU/EN/KZ) на ключевых сценариях
- [ ] Логи не содержат секретов / PII
- [ ] Существующие сценарии (FAQ, верификация, админ-команды) не сломаны

**После выпуска:**
- [ ] Мониторинг `query_logs` за первые сутки: распределение `clarification_rounds`, средняя длина ответов LLM, доля `feedback = -1`
- [ ] Решение о переходе к следующему этапу — после 3-5 дней стабильной работы

---

## Что НЕ входит в этот план (явно)

- Персистентное FSM-хранилище (SQLite/Redis) — отложено.
- Структурированный `user_profile` и поля `enriched_query` / `user_profile` / `reranker_scores` в `query_logs` — только `clarification_rounds`.
- Кастомные веса факторов как админ-настройка — на Фазе 4 веса захардкожены (из описания).
- Метрики «сколько раз LLM решил остановить диалог досрочно» — добавим, если на Этапе 1-2 встанет вопрос калибровки.
