# Промпты для внедрения функционала «Уточняющий диалог перед RAG-поиском»

> Этот файл — последовательность промптов, которые можно подавать Claude Code (или другому AI-агенту) **по одному за раз**, в указанном порядке. Каждый промпт самодостаточен: содержит контекст, цель, файлы и критерии готовности, чтобы агент мог стартовать с нуля без памяти предыдущих сессий.
>
> Источники: [dialog_feature_description.md](dialog_feature_description.md), [dialog_feature_implementation_plan.md](dialog_feature_implementation_plan.md).
>
> **Как пользоваться:**
> 1. Каждый промпт — отдельная задача (отдельный PR или коммит).
> 2. После выполнения промпта — ручное тестирование по критериям в конце промпта.
> 3. Переход к следующему — только когда текущий этап стабилен.

---

## ПРОМПТ 0 — Подготовка инфраструктуры пакета `rag/dialog/`

```
Контекст: проект AITU_chatbot — Telegram-бот для университета, отвечающий на вопросы
через RAG (Qdrant + LLM). Сейчас pipeline в bot/handlers/user.py:280-342:
text → retriever.search → generator.generate. Мы внедряем фичу «Уточняющий диалог»:
перед поиском бот может задать 1–3 уточняющих вопроса. Полное описание —
dialog_feature_description.md; план — dialog_feature_implementation_plan.md.

Задача:
Создать каркас нового пакета rag/dialog/ — пока без бизнес-логики, только структура
и общие утилиты. Это подготовка для следующих этапов.

Сделать:
1. Создать директорию rag/dialog/ со следующими файлами:
   - __init__.py (пустой)
   - prompts.py — модуль для системных промптов LLM (пока пустой, только docstring)
   - classifier.py — заглушка с функцией classify_intent(question: str) -> dict
     возвращающей {"needs_clarification": False, "reason": "stub"} (всегда быстрый путь)
   - question_gen.py — заглушка
   - enricher.py — заглушка
2. В каждом файле — модульный docstring, объясняющий назначение.
3. Импорт LLM-клиента сделать через _make_llm_client из rag/generator.py:124-133 —
   НЕ дублировать его логику.
4. Запустить бота локально (`python -m bot.main`), убедиться что ничего не сломалось,
   что классификатор-заглушка не вызывается (она ещё нигде не подключена).

Критерии готовности:
- Структура каталогов создана.
- `python -c "from rag.dialog import classifier, question_gen, enricher, prompts"` отрабатывает без ошибок.
- Бот запускается, существующие сценарии (вопрос → ответ, FAQ) работают как прежде.

НЕ делать в этом промпте: подключать классификатор к user.py, писать реальные промпты,
менять схему БД.
```

---

## ПРОМПТ 1 — Этап 1: Классификация намерения

```
Контекст: пакет rag/dialog/ уже создан (см. ПРОМПТ 0). Сейчас все запросы идут в
retriever.search напрямую через bot/handlers/user.py:280-342. Нам нужно перед поиском
вызывать LLM-классификатор, который решает: запрос конкретный (искать сразу) или
расплывчатый (нужны уточнения). Описание классификации — Фаза 1 в dialog_feature_description.md.

На этом этапе диалог ещё не реализован: при «расплывчатом» вердикте бот пока всё равно
идёт по старому пути (но логирует решение классификатора). Это нужно, чтобы откалибровать
классификатор на реальных запросах перед тем, как блокировать пользователя уточнениями.

Задача:
1. В rag/dialog/prompts.py добавить константу CLASSIFY_SYSTEM_PROMPT — системный промпт
   на английском, инструктирующий LLM:
   - решить, нужно ли уточнение, по двум критериям:
     * запрос конкретный и самодостаточный → нет
     * запрос расплывчатый или многозначный → да
   - примеры: см. Фаза 1 в dialog_feature_description.md
   - вернуть ТОЛЬКО JSON: {"needs_clarification": bool, "reason": "specific"|"vague_topic"|"ambiguous"}
   - язык запроса любой (RU/EN/KZ) — промпт language-agnostic.

2. В rag/dialog/classifier.py реализовать:
   - TypedDict ClassificationResult с полями needs_clarification, reason
   - async def classify_intent(question: str) -> ClassificationResult
     * один LLM-вызов через _make_llm_client() из rag/generator.py
     * model = settings.llm_model, temperature = 0, max_tokens = 50
     * парсинг JSON из ответа (искать первую { ... } как в rag/generator.py:177-184)
     * на ЛЮБУЮ ошибку (network, parse, timeout) — fallback:
       ClassificationResult(needs_clarification=False, reason="error")
       и logger.warning с трейсом.

3. В bot/handlers/user.py:281 (handle_question) после rate-limit проверки и ДО
   "🔍 Ищу информацию..." добавить:
   ```python
   intent = await classify_intent(question)
   logger.info("Intent classification: q='%.60s' result=%s", question, intent)
   ```
   На текущем этапе НЕ менять поведение в зависимости от intent — только логировать.

4. Прогнать через бота ~10 «конкретных» и ~10 «расплывчатых» запросов вручную,
   собрать логи (logs/ или stdout). Сравнить с ожиданиями. Если ошибок > 30% —
   подкрутить промпт в prompts.py и повторить. Зафиксировать финальную версию промпта.

Критерии готовности:
- classify_intent возвращает корректные JSON-результаты.
- На «Какой штраф за академическую задолженность?» → needs_clarification=False.
- На «Что мне делать?» / «Расскажи про правила» → needs_clarification=True.
- Бот продолжает работать как раньше (классификация — пока read-only).
- Метрика точности на 20 тестовых запросах: ≥ 85%.

Файлы:
- rag/dialog/prompts.py (создан в ПРОМПТЕ 0)
- rag/dialog/classifier.py (создан в ПРОМПТЕ 0)
- bot/handlers/user.py (модифицировать handle_question)

НЕ делать: запускать диалог уточнений, менять схему БД, трогать retriever/generator.
```

---

## ПРОМПТ 2 — Этап 2 (часть А): FSM-инфраструктура и роутер диалога

```
Контекст: классификатор намерения работает и логирует решения (см. ПРОМПТ 1).
Теперь нужно реализовать сам уточняющий диалог: при `needs_clarification=True` бот
задаёт вопросы через inline-кнопки, держит состояние FSM, и через 1–3 раунда
переходит к поиску.

Хранилище FSM — MemoryStorage (как для верификации email в bot/auth/handler.py).
Структура состояний и хендлеров описана в Этап 2 плана dialog_feature_implementation_plan.md.

На этой части (А) реализуем только инфраструктуру: состояния, хендлеры, переход в
диалог. Сама генерация вопросов через LLM — в следующей части (Б).

Задача:
1. Проверить, что Dispatcher в bot/main.py инициализируется с MemoryStorage. Если нет —
   добавить (FSMContext должен работать на всех роутерах). Если уже добавлен для auth —
   ничего не делать.

2. Создать bot/handlers/dialog_states.py:
   ```python
   from aiogram.fsm.state import State, StatesGroup
   class ClarifyDialog(StatesGroup):
       waiting_for_answer = State()
   ```

3. Создать bot/handlers/dialog.py:
   - Router с именем router (как в других хендлерах).
   - Заглушка функции `_generate_question(...)` пока возвращает фиксированный вопрос:
     `{"question": "Уточните, пожалуйста, тему", "options": ["Академическая", "Административная"], "stop": False}`.
   - async def start_clarification_dialog(message, original_query, state):
     * вызывает _generate_question (заглушку)
     * сохраняет в state: {"original_query": ..., "rounds_done": 0, "answers": []}
     * переводит в ClarifyDialog.waiting_for_answer
     * отправляет вопрос с inline-кнопками (см. clarify_keyboard ниже)
   - async def _ask_next(message, state) — следующий вопрос или переход к поиску.
   - async def _proceed_to_search(message, state) — пока ВРЕМЕННАЯ реализация:
     взять original_query из state, вызвать тот же код что в handle_question (импорт
     retriever/generator из bot/handlers/user.py — допустимо). На Этапе 3 заменим
     на обогащённый запрос.
   - clarify_keyboard(round_no, options) → InlineKeyboardMarkup:
     * кнопки `clarify:<round>:<option_index>` (только индекс, текст в state)
     * последняя строка — кнопка "⏭ Пропустить" → `clarify:<round>:skip`.

4. Три хендлера в dialog.py (порядок регистрации важен):
   - @router.callback_query(F.data.startswith("clarify:")) — парсит round и option_index/skip,
     достаёт текст из state["last_options"], сохраняет ответ в state["answers"],
     инкрементит rounds_done, отвечает callback.answer() и вызывает _ask_next.
   - @router.message(Command("skip"), ClarifyDialog.waiting_for_answer) — помечает раунд
     как пропущенный, вызывает _ask_next.
   - @router.message(F.text, ClarifyDialog.waiting_for_answer) — свободный текст.
     Если текст совпадает (case-insensitive, strip) с одним из:
     ["не знаю", "неважно", "не важно", "idk", "whatever", "pass"] — завершить диалог
     (_proceed_to_search). Иначе — записать как ответ и вызвать _ask_next.

5. Лимит раундов: если rounds_done >= 3 → _proceed_to_search независимо от пути.

6. Зарегистрировать router в bot/main.py ДО bot/handlers/user.py router, чтобы
   F.text-хендлер диалога перехватывал ответы в состоянии раньше общего text-handler.

7. В bot/handlers/user.py:handle_question — теперь подключить классификатор «по-настоящему»:
   ```python
   intent = await classify_intent(question)
   if intent["needs_clarification"]:
       from bot.handlers.dialog import start_clarification_dialog
       await start_clarification_dialog(message, question, state)
       return
   # дальше — текущая логика
   ```
   В сигнатуру handle_question добавить параметр `state: FSMContext` (aiogram сам
   проставит при include_router).

Критерии готовности:
- Расплывчатый запрос «Что мне делать?» → бот отправляет заглушку вопроса с двумя кнопками.
- Нажатие кнопки → бот спрашивает второй (заглушечный) вопрос.
- После 3 раундов → бот выполняет обычный поиск (по original_query, без обогащения — пока ок).
- /skip → сразу переход к поиску.
- «не знаю» текстом → переход к поиску.
- Конкретный запрос («Как оформить академический отпуск?») → классификатор не триггерит,
  идёт прямой поиск (текущее поведение).
- Существующие сценарии (FAQ, верификация, админка) не сломаны.

НЕ делать в этой части: реальный LLM для генерации вопросов, обогащение запроса,
миграции БД.
```

---

## ПРОМПТ 3 — Этап 2 (часть Б): LLM-генерация уточняющих вопросов

```
Контекст: FSM-инфраструктура диалога готова (ПРОМПТ 2). Сейчас вопросы — захардкоженная
заглушка. Нужно заменить её на динамическую генерацию через LLM, опираясь на исходный
запрос, список доступных документов и уже полученные ответы.

Задача:
1. В rag/dialog/prompts.py добавить QUESTION_GEN_SYSTEM_PROMPT — системный промпт,
   инструктирующий LLM:
   - На основе original_query, списка документов (doc_title/section_title) и истории
     уже заданных уточнений сгенерировать ОДИН следующий уточняющий вопрос.
   - Типы уточнений (из dialog_feature_description.md): тема / статус пользователя /
     контекст / документ.
   - Формат ответа — строгий JSON:
     {"question": str, "options": [str, ...], "stop": bool}
     * options: 2–4 варианта на языке original_query, либо пустой массив если уместен
       только свободный ввод.
     * stop=true если контекста уже достаточно (например: тема ясна и спрашивать больше
       нечего, или раунд 3 — последний).
   - Промпт language-aware: вопрос и опции — на ЯЗЫКЕ original_query.

2. В rag/dialog/question_gen.py реализовать:
   - TypedDict ClarificationQuestion (question: str, options: list[str], stop: bool).
   - async def next_clarification(state_data: dict, available_docs: list[dict]) -> ClarificationQuestion
     * state_data: {original_query, rounds_done, answers}
     * available_docs: упрощённый список [{"doc_title": ..., "section_title": ...}, ...]
       (см. ниже про источник).
     * Один LLM-вызов (model=settings.llm_model, temperature=0.2, max_tokens=400).
     * Парсинг JSON; на ошибку → {"question": "", "options": [], "stop": True}
       (это безопасно: вызывающий код в таком случае перейдёт к поиску).
     * Лимит options: max 4 (отрезать лишние).
     * Длина каждой option ≤ 50 символов (отрезать с многоточием) — Telegram inline
       button label.

3. Кеширование списка документов:
   - В rag/dialog/question_gen.py добавить in-memory кеш:
     ```python
     _docs_cache = {"items": None, "expires_at": 0.0}
     _DOCS_TTL = 600  # 10 минут
     async def _get_cached_docs(retriever) -> list[dict]:
         # вернуть кеш или обновить через retriever.get_all_documents()
     ```
   - Источник — Retriever.get_all_documents() из rag/retriever.py:178-213. Для
     section_title — взять список уникальных section_title через ОДНОКРАТНЫЙ scroll
     Qdrant с with_payload=["doc_title","section_title"] (не вызывать get_all_documents
     дважды). Если получается дорого — на MVP вернуть только doc_title без section_title,
     это допустимо.

4. В bot/handlers/dialog.py:
   - Заменить заглушку `_generate_question` на вызов `next_clarification`.
   - Прокинуть `_retriever` (взять из bot/handlers/user.py или создать локальный
     экземпляр — НЕ дублировать клиент Qdrant, переиспользовать тот же объект; если
     это создаёт циклический импорт — создать отдельный модуль bot/retriever_singleton.py
     с одним глобальным Retriever).
   - Сохранять в state["last_options"]: list[str] перед отправкой каждого вопроса,
     чтобы callback мог восстановить текст по индексу.

5. Защита от пустых вопросов: если next_clarification вернул `question=""` или `stop=True`
   на первом же раунде → НЕ показывать пустое сообщение, сразу _proceed_to_search.

Критерии готовности:
- «Какие есть льготы?» → бот задаёт осмысленный вопрос (например, «Вас интересует
  стипендия, общежитие или питание?») с релевантными кнопками.
- «Расскажи про правила» → бот спрашивает категорию правил.
- Кнопки на казахском, если original_query на казахском (проверить хотя бы один пример).
- После 1–3 раундов диалог завершается (либо LLM ставит stop=true, либо достигнут
  лимит).
- При недоступности LLM (отрубить интернет на 5 сек) → диалог корректно
  деградирует в поиск без падения.

НЕ делать: обогащение запроса, миграции БД.
```

---

## ПРОМПТ 4 — Этап 3: Обогащение запроса и логирование

```
Контекст: уточняющий диалог работает, собирает ответы пользователя (ПРОМПТЫ 2-3).
Сейчас после диалога в retriever отправляется ОРИГИНАЛЬНЫЙ запрос — это сводит на нет
весь смысл уточнений. Нужно: (а) обогащать запрос ответами пользователя ДЛЯ поиска,
(б) в LLM-генератор всё равно передавать оригинальный вопрос (чтобы ответ звучал
естественно), (в) логировать количество раундов уточнений.

Задача:
1. В rag/dialog/prompts.py добавить ENRICH_SYSTEM_PROMPT — системный промпт:
   - На вход: original_query + список (вопрос, ответ).
   - На выход: ОДНА строка-поисковый-запрос, объединяющая исходный запрос и контекст
     из ответов, на ЯЗЫКЕ original_query.
   - Пример из dialog_feature_description.md:
     original: «Какие есть льготы?»
     answers: статус=бакалавр, тема=общежитие, период=текущий год
     output: «льготы на проживание в общежитии для студентов бакалавриата 2024-2025»
   - Никаких JSON, кавычек, объяснений — только сама строка.

2. В rag/dialog/enricher.py:
   - async def enrich_query(original: str, answers: list[dict]) -> str
     * answers: [{"question": str, "answer": str}, ...]
     * Если answers пуст — вернуть original без LLM-вызова.
     * Один LLM-вызов (temperature=0.1, max_tokens=200).
     * На любую ошибку → return original (fallback graceful).

3. Миграция БД: добавить колонку clarification_rounds в query_logs:
   - Найти место инициализации таблицы query_logs (вероятно в bot/main.py или
     в bot/handlers/feedback.py; искать `CREATE TABLE … query_logs`).
   - Рядом с CREATE TABLE добавить идемпотентную миграцию:
     ```python
     try:
         await db.execute("ALTER TABLE query_logs ADD COLUMN clarification_rounds INTEGER DEFAULT 0")
     except aiosqlite.OperationalError:
         pass  # column already exists
     ```
   - Если CREATE TABLE создаётся «с нуля» — также включить новую колонку в CREATE.

4. Расширить log_query в bot/handlers/feedback.py:22-44:
   - Добавить параметр `clarification_rounds: int = 0`.
   - Включить в INSERT (новая колонка).

5. В bot/handlers/dialog.py:_proceed_to_search:
   ```python
   data = await state.get_data()
   original_query = data["original_query"]
   answers = data.get("answers", [])
   rounds = data.get("rounds_done", 0)

   enriched = await enrich_query(original_query, answers) if answers else original_query
   chunks = await _retriever.search(enriched)
   result = await _generator.generate(original_query, chunks)  # ← original, не enriched
   log_id = await log_query(
       user_id=...,
       query=original_query,
       detected_lang=result["detected_lang"],
       chunks=chunks,
       answer=result["answer"],
       sources=result["sources"],
       clarification_rounds=rounds,
   )
   # отправить ответ пользователю (см. bot/handlers/user.py:320-342 — реиспользовать
   # _build_sources_text, _disclaimer, _build_keyboard, send_long_message)
   await state.clear()
   ```

6. Также в обычной ветке (handle_question без диалога) добавить
   `clarification_rounds=0` в log_query, чтобы все запросы имели поле.

Критерии готовности:
- «Какие есть льготы?» → 2 уточнения → в логах SELECT clarification_rounds FROM query_logs
  ORDER BY id DESC LIMIT 1 показывает 2.
- enriched_query в логах debug-уровня визуально соответствует ожиданиям
  (можно добавить временный logger.info, удалить после проверки).
- Ответ LLM звучит как ответ на ОРИГИНАЛЬНЫЙ вопрос пользователя, а не на
  «машинный» расширенный.
- A/B вручную: тот же расплывчатый вопрос с уточнениями возвращает более
  релевантные источники, чем без.
- Старые логи без новой колонки работают (миграция не падает на существующей БД).

НЕ делать: многофакторный скоринг, LLM-реранкер, user_profile.
```

---

## ПРОМПТ 5 — Стабилизационный прогон перед Этапом 4

```
Контекст: фичу «Уточняющий диалог» (Этапы 1–3) выкатили в работу. Перед тем как
браться за многофакторный скоринг и LLM-реранкер (Этап 4 плана), нужна
стабилизационная проверка.

Задача (диагностика, не написание кода):
1. Снять метрики за последние N дней работы бота (или с момента выкатки):
   - Доля запросов с clarification_rounds=0 vs >0 (цель: 60–70% без уточнений).
   - Средняя длина ответа LLM в обеих группах.
   - Доля отрицательного feedback (`feedback=-1`) в обеих группах — стало хуже или лучше?
   SQL для query_logs (sqlite, settings.sqlite_db_path):
   ```sql
   SELECT
     CASE WHEN clarification_rounds = 0 THEN 'no_clarify' ELSE 'with_clarify' END AS path,
     COUNT(*) AS n,
     AVG(answer_length) AS avg_len,
     SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS bad_rate
   FROM query_logs GROUP BY path;
   ```

2. Прочитать ~20 свежих query_logs с clarification_rounds > 0 (через
   `SELECT query, answer FROM query_logs WHERE clarification_rounds > 0
   ORDER BY id DESC LIMIT 20;`). Глазами оценить:
   - Уточнения уместны?
   - Финальный ответ улучшился по сравнению с тем, что было бы без диалога?
   - Где видны систематические ошибки классификатора (false positive / false negative)?

3. Сформировать короткий отчёт (markdown, до 1 страницы): что работает, что плохо,
   рекомендации до старта Этапа 4 (например: подкрутить промпт классификатора,
   изменить лимит раундов с 3 на 2, и т.п.).

Критерии готовности:
- Отчёт в файле dialog_feature_stabilization_report.md создан.
- В нём — таблица метрик и качественные наблюдения.
- Решение «идём в Этап 4» / «сначала чиним X» — задокументировано.

НЕ делать: писать код.
```

---

## ПРОМПТ 6 — Этап 4 (часть А): Извлечение профиля пользователя

```
Контекст: MVP диалога (Этапы 1–3) работает в проде, стабилизация пройдена (ПРОМПТ 5).
Теперь начинаем Фазу 4 — многофакторный скоринг чанков. Первый шаг: извлекать из
ответов пользователя структурированный профиль (а не только текстовое обогащение).

Профиль (из dialog_feature_description.md, раздел Фаза 3):
- topics: list[str]
- user_type: Optional[str]  — "бакалавр" | "магистрант" | "сотрудник" | None
- document_hints: list[str]  — упоминаемые названия документов
- temporal_context: Optional[str]  — "текущий семестр", "2024", etc.

Задача:
1. В rag/dialog/prompts.py добавить EXTRACT_PROFILE_SYSTEM_PROMPT — инструкция LLM
   вернуть JSON с указанной схемой; пустые поля → null или [].

2. В rag/dialog/enricher.py добавить:
   - TypedDict UserProfile с теми же 4 полями.
   - async def extract_profile(original: str, answers: list[dict]) -> UserProfile
     * Один LLM-вызов (temperature=0.0, max_tokens=300).
     * Строгий JSON-парсинг (как в classify_intent).
     * На ошибку → UserProfile(topics=[], user_type=None, document_hints=[], temporal_context=None).

3. В bot/handlers/dialog.py:_proceed_to_search:
   - После enrich_query параллельно (asyncio.gather) вызвать extract_profile.
   - Профиль — пока НЕ передаётся в retriever (это в части Б). Просто сохранить
     в локальной переменной и логировать `logger.info("Extracted profile: %s", profile)`.

4. Тесты вручную:
   - «Какие льготы для магистрантов?» → 1–2 уточнения → profile.user_type=="магистрант".
   - «Правила проживания в общежитии» → profile.topics включает «общежитие»,
     profile.document_hints может ссылаться на правила внутреннего распорядка
     (если такие документы есть в Qdrant).

Критерии готовности:
- Профиль извлекается, в логах виден корректный JSON.
- На неоднозначных запросах поля null / пустые массивы (без галлюцинаций).
- Производительность: общее время от первого сообщения пользователя до ответа
  выросло не более чем на 1.5 сек по сравнению с MVP.

НЕ делать: использовать профиль в retriever, реранкер, новые колонки в query_logs.
```

---

## ПРОМПТ 7 — Этап 4 (часть Б): Многофакторный скоринг в Retriever

```
Контекст: профиль пользователя извлекается (ПРОМПТ 6). Теперь подключаем
многофакторный скоринг чанков. Веса факторов — из dialog_feature_description.md,
Фаза 4.

Задача:
1. В rag/retriever.py добавить метод:
   async def search_with_profile(self, enriched_query: str, profile: dict, k: int | None = None) -> list[dict]

   Алгоритм:
   a. Поднять кандидатов: ровно как в search(), но limit = settings.top_k * 5
      (вместо * 3), with_vectors=True.
   b. Для каждого кандидата вычислить 6 факторов на интервале [0, 1]:
      - semantic_sim (0.35): нормализованный point.score (cosine, уже [0..1]).
      - user_type_match (0.20):
        * Если profile["user_type"] is None → 0.5 (нейтрально).
        * Иначе: 1.0 если ключевое слово ("бакалавр"/"магистр"/...) встречается
          в chunk["text"] или section_title (case-insensitive); 0.0 иначе.
      - doc_match (0.15):
        * Если profile["document_hints"] пуст → 0.5.
        * Иначе: max(SequenceMatcher.ratio(hint.lower(), doc_title.lower())
          for hint in hints) — fuzzy match.
      - position_bonus (0.10):
        * 1.0 если chunk.section_title не пуст И в первых 1500 символах текста
          встречается слово из section_title; 0.5 иначе.
        * Эвристика проста — на практике достаточно.
      - recency (0.10):
        * Парсить chunk["uploaded_at"] (ISO timestamp).
        * Возраст в днях. score = exp(-age_days / 365) — год даёт ~0.37.
        * При отсутствии даты → 0.5.
      - section_sim (0.10):
        * cosine(embedder.embed_query(section_title), enriched_query_vec).
        * Кеш по section_title (in-memory dict, ключ — section_title).
        * При пустом section_title → 0.5.
   c. final_score = sum(weight_i * factor_i)
   d. Сортировать по final_score, вернуть top-k (k = settings.top_k или переданный).
   e. К каждому возвращаемому чанку добавить ключ "factor_scores": dict с шестью
      значениями — для DEBUG-логов.

2. В bot/handlers/dialog.py:_proceed_to_search:
   - Если profile содержит хоть одно непустое поле → использовать
     _retriever.search_with_profile(enriched, profile); иначе fallback на
     _retriever.search(enriched).
   - logger.debug("Top chunk factor scores: %s", chunks[0].get("factor_scores"))

3. Производительность: пересчёт факторов — O(n) по чанкам без LLM, дополнительный
   embedder-вызов только для уникальных section_title в этом запросе (с кешом).
   Сумма должна добавить < 1 сек.

Критерии готовности:
- На запросе с явным user_type («Какие льготы для магистрантов?») в top-3 чанков
  чаще встречается слово «магистр», чем при чистом cosine.
- factor_scores в логах выглядят разумно (нет zero-divide, NaN, ничего > 1).
- Старые запросы без диалога (clarification_rounds=0) проходят через старый search() —
  не влияем на этот путь.

НЕ делать: LLM-реранкер, колонку reranker_scores в БД.
```

---

## ПРОМПТ 8 — Этап 4 (часть В): LLM-реранкер финальной выдачи

```
Контекст: многофакторный скоринг работает (ПРОМПТ 7). Финальный шаг — LLM-реранкинг
top-10 кандидатов, чтобы отсеять ложные совпадения по похожим словам с разным смыслом.

Задача:
1. Создать rag/dialog/reranker.py:
   async def rerank_chunks(query: str, chunks: list[dict], k: int = 5) -> list[dict]

   - Вход: ранжированный список из search_with_profile (до 10 чанков).
   - Если len(chunks) <= k → вернуть как есть (без LLM-вызова).
   - Один LLM-вызов с промптом из prompts.py (RERANK_SYSTEM_PROMPT):
     * на вход — пронумерованный список чанков (заголовок + первые 300 символов текста);
     * вернуть JSON: {"order": [int, ...], "reasons": [str, ...]}, где order — новый
       порядок индексов, reasons — однострочные объяснения почему чанк релевантен.
   - Переставить chunks по `order`, обрезать до k.
   - reasons → DEBUG-лог (не пользователю).
   - На ошибку парсинга → вернуть исходный top-k без изменений.

2. В rag/dialog/prompts.py добавить RERANK_SYSTEM_PROMPT (на английском, инструкция
   возвращать строгий JSON).

3. В bot/handlers/dialog.py:_proceed_to_search:
   ```python
   chunks = await _retriever.search_with_profile(enriched, profile)  # top-10
   chunks = await rerank_chunks(original_query, chunks, k=settings.top_k)
   ```
   Использовать original_query (не enriched) — реранкер оценивает релевантность
   с точки зрения пользователя.

4. Производительность: один доп. LLM-вызов; max_tokens=500, temperature=0.
   На стороне UI — пользователь уже видит «Ищу информацию...», задержка приемлема.

5. Метрика: после выкатки сравнить feedback rate (-1 / total) до и после на
   запросах с clarification_rounds > 0. Цель — снижение не менее чем на 10%.

Критерии готовности:
- На запросе, где старый поиск возвращал «правильный» чанк на 6-м месте, реранкер
  ставит его в top-3 (вручную проверить 5 примеров).
- При недоступности LLM (timeout) → реранкер не валит запрос, возвращает старый порядок.
- Логи содержат `reasons` для каждого чанка в верхушке выдачи (для дебага).

НЕ делать: персистить factor_scores / reasons в БД (отложено по решению из плана).
```

---

## Финальные замечания

- **Порядок выполнения промптов жёсткий.** Каждый зависит от артефактов предыдущего.
- Между ПРОМПТАМИ 4 и 6 — обязательно ПРОМПТ 5 (стабилизация), иначе ранние ошибки
  классификатора смешаются с эффектом реранкера.
- Если по ходу выясняется, что какой-то Этап не нужен (например, multifactor scoring
  не даёт прироста по метрикам в ПРОМПТЕ 5) — отказаться, а не «доделывать ради
  плана».
- Все промпты содержат секцию «НЕ делать» — это защита от scope creep при автономном
  исполнении.
