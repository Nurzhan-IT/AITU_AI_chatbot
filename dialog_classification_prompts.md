# Серия промптов: внедрение улучшений классификации `needs_clarification`

Готовые промпты для пошагового применения изменений из
[dialog_classification_improvements.md](dialog_classification_improvements.md).
Каждый промпт самодостаточен — копируется в агент целиком и выполняется как
отдельная задача.

## Как пользоваться

- Все пути в промптах **относительно** `AITU_AI_chatbot/` (там, где лежит `config.py`).
- Выполняйте промпты **по порядку фаз** — между ними есть зависимости (отмечены
  в каждом промпте полем «Зависит от»).
- После каждого промпта прогоняйте бота вручную (`python -m bot.main`) — в проекте
  нет тестов и линтера, поэтому проверка только ручная.
- Каждый промпт = один атомарный коммит. Не смешивайте несколько промптов в одном.
- Промпты фазы B содержат не только код, но и операционные шаги (накопление
  запросов, ручная разметка) — они помечены `[операционное]`.

## Карта зависимостей

```
Фаза A  (A1…A8)   — независимы друг от друга, можно в любом порядке
   ↓
Фаза B  (B1…B6)   — B4/B5/B6 нужны как вход для фазы C
   ↓
Фаза C  (C1…C5)   — C1 зависит от B4 (калибровка порогов); C4 зависит от C1
   ↓
Фаза D  (D1…D3)   — D1/D2 зависят от C1 (probe retrieval даёт top-15)
   ↓
Фаза E  (E1…E4)   — опционально, независимы
```

---

# Фаза A. Quick wins

---

## A1 — Сократить `max_tokens` классификатора (§2.1)

```text
Контекст: Telegram-бот AITU, RAG-пайплайн. Файл rag/dialog/classifier.py.

Задача: в classify_intent (rag/dialog/classifier.py:38) уменьшить max_tokens
с 1000 до 50. Ответ классификатора — однострочный JSON (~50 токенов),
буфер на 1000 токенов избыточен и зря увеличивает latency/стоимость.

Критерии готовности:
- max_tokens=50 в вызове client.chat.completions.create.
- Прогнать classify_intent на 3–4 запросах разной длины — JSON по-прежнему
  парсится полностью, обрезания ответа нет.
```

---

## A2 — Добавить `confidence` в выход классификатора (§2.2)

```text
Контекст: Telegram-бот AITU. Файлы rag/dialog/classifier.py, rag/dialog/prompts.py.

Задача: добавить поле confidence в результат классификатора.
1. В CLASSIFY_SYSTEM_PROMPT (rag/dialog/prompts.py) дописать в формат вывода
   поле confidence (число 0.0–1.0):
   {"needs_clarification": true, "reason": "ambiguous", "confidence": 0.0–1.0}
2. В classify_intent (rag/dialog/classifier.py) распарсить confidence,
   добавить его в TypedDict ClassificationResult и в возвращаемый результат.
   При отсутствии/невалидном значении — confidence=0.5.
3. Завести в config.py два порога: CLASSIFY_CONF_HIGH=0.7, CLASSIFY_CONF_LOW=0.4.
   Использовать пока только для логирования смысла (>=0.7 → диалог,
   0.4–0.7 → пограничный, <0.4 → прямой поиск). Менять поведение хендлера
   в этом промпте НЕ нужно.

Важная оговорка из дизайн-документа: self-reported confidence от LLM при
temperature=0 калибруется плохо — это эвристика, временная мера. Её вытеснит
статистический триаж по score-распределению (фаза C). НЕ вкладывайтесь в
тонкую настройку порогов — оставьте round-числа.

Критерии готовности:
- classify_intent возвращает confidence; ClassificationResult обновлён.
- Бот запускается, классификатор не падает на отсутствии поля.
```

---

## A3 — Skip-слова на казахском (§2.3)

```text
Контекст: Telegram-бот AITU, уточняющий диалог. Файл bot/handlers/dialog.py:32.

Задача: в множество _STOP_WORDS (bot/handlers/dialog.py:32) добавить казахские
варианты «не знаю / неважно / пропустить»:
  "білмеймін", "маңызды емес", "өткізіп жіберу".
Стоп-слова сравниваются в нижнем регистре (raw.lower() in _STOP_WORDS) —
добавляемые строки тоже должны быть в нижнем регистре.

Критерии готовности:
- _STOP_WORDS содержит 3 новых KZ-слова.
- В уточняющем диалоге ввод «білмеймін» завершает диалог и уходит в поиск
  (как «не знаю»).
```

---

## A4 — Инвалидация doc-кэша при загрузке документа (§2.4)

```text
Контекст: Telegram-бот AITU. Файлы rag/dialog/question_gen.py, ingestion/ingest.py.

Задача: кэш списка документов _docs_cache живёт в rag/dialog/question_gen.py
с TTL 10 минут. После загрузки нового документа генератор уточняющих вопросов
до 10 минут «не видит» новое название.

1. В rag/dialog/question_gen.py добавить ПУБЛИЧНУЮ функцию
   invalidate_docs_cache() — она сбрасывает expires_at кэша в 0.
2. В ingestion/ingest.py после успешной загрузки/индексации документа вызвать
   invalidate_docs_cache(). Импортировать функцию ПО МЕСТУ (внутри функции),
   чтобы не создавать циклический импорт ingest ↔ dialog.

Запрещено: мутировать приватный _docs_cache снаружи модуля question_gen.py —
только через публичную invalidate_docs_cache().

Критерии готовности:
- invalidate_docs_cache() экспортируется из question_gen.py.
- ingest.py вызывает её после успешного upsert в Qdrant.
- После загрузки документа следующий _get_cached_docs возвращает свежий список.
```

---

## A5 — Объединить `enrich_query` + `extract_profile` в один LLM-вызов (§2.5)

```text
Контекст: Telegram-бот AITU, уточняющий диалог.
Файлы rag/dialog/enricher.py, bot/handlers/dialog.py.

Задача: сейчас enrich_query и extract_profile (rag/dialog/enricher.py) читают
один и тот же вход (original + answers) и гоняют LLM параллельно через
asyncio.gather (bot/handlers/dialog.py:143). Объединить в ОДИН LLM-вызов,
возвращающий {"enriched": "...", "profile": {...}}.

Нюансы реализации:
- extract_profile сейчас возвращает JSON, а enrich_query — ЧИСТЫЙ ТЕКСТ
  (его промпт требует «Output PLAIN TEXT only»). Объединённый вызов переводит
  вывод enriched в поле JSON — обновите промпт и парсинг соответственно.
- Обновите ветку fallback: если LLM вернул невалидный JSON, fallback должен
  отдавать enriched=original_query и пустой profile (как раньше extract_profile).
- В bot/handlers/dialog.py:143 заменить asyncio.gather на один await новой
  функции. Импорт asyncio оставить, если он используется ещё где-то.

Цель — экономия ~1 секунды на каждый уточнённый диалог.

Критерии готовности:
- Новая функция (например enrich_and_profile) делает 1 LLM-вызов.
- bot/handlers/dialog.py больше не вызывает enrich_query и extract_profile
  параллельно.
- Уточнённый диалог по-прежнему отдаёт корректный enriched + profile;
  fallback при битом JSON не роняет хендлер.
```

---

## A6 — Доменные few-shot примеры в промпте классификатора (§2.6)

```text
Контекст: Telegram-бот AITU. Файл rag/dialog/prompts.py:28-35.

Задача: в few-shot блок CLASSIFY_SYSTEM_PROMPT (rag/dialog/prompts.py, строки
~28-35) добавить 4–6 примеров ИЗ РЕАЛЬНОГО КОРПУСА AITU:
- реальные названия документов (правила общежития, академическая политика,
  финансовый регламент и т.п.);
- реальные user-type-зависимые вопросы (бакалавр/магистрант, очное/заочное).

Перед написанием примеров посмотрите фактический список документов: запустите
`python -m ingestion.ingest --dir pdfs/` НЕ нужно — просто загляните в каталог
pdfs/ и/или таблицу file_history в data/bot.db, чтобы взять настоящие названия.
Каждый пример: запрос → ожидаемый JSON ({needs_clarification, reason,
confidence}). Покройте оба класса: specific и vague_topic/ambiguous.

Критерии готовности:
- В промпте 4–6 новых примеров с реальными названиями документов AITU.
- Формат примеров совпадает с форматом вывода из A2 (с полем confidence).
- Классификатор по-прежнему возвращает валидный JSON на контрольных запросах.
```

---

## A7 — Логировать `classification_reason` в `query_logs` (§2.7)

```text
Контекст: Telegram-бот AITU. Работа в трёх местах.
Файлы: duplicate_detection/db.py, bot/handlers/feedback.py, bot/handlers/user.py,
bot/handlers/dialog.py, bot/handlers/dialog_states.py.

Задача: протащить reason из классификатора в таблицу query_logs.

1. Схема: в duplicate_detection/db.py добавить колонку classification_reason
   в query_logs. Использовать уже имеющийся там идемпотентный паттерн миграции
   (ALTER TABLE ... в try/except OperationalError) — не ломать существующую БД.

2. log_query: в bot/handlers/feedback.py добавить параметр
   classification_reason (default None) в сигнатуру log_query и в INSERT.

3. Прямой путь: в bot/handlers/user.py получить reason из classify_intent
   и передать его в log_query.

4. Уточнённый путь: в bot/handlers/dialog.py исходный intent/reason в
   _proceed_to_search сейчас НЕДОСТУПЕН. Сохранить reason в FSM-состоянии
   (state.update_data) в start_clarification_dialog и прочитать его в
   _proceed_to_search, затем передать в log_query.

Назначение: построить evaluation-set из реальных запросов (нужно для фазы B).

Критерии готовности:
- Колонка classification_reason появляется в query_logs (новая и существующая БД).
- Для прямых и для уточнённых запросов reason записывается в БД.
- log_query обратно совместим (параметр опциональный).
```

---

## A8 — Порог силы профиля для дорогого пути (§4.9, узкое место №9)

```text
Контекст: Telegram-бот AITU. Файл bot/handlers/dialog.py:158.

Задача: profile_has_signal (bot/handlers/dialog.py:158) сейчас — простой OR:
любой один непустой слот включает дорогой путь search_with_profile + LLM rerank.
Заменить на оценку СИЛЫ профиля.

Дорогой путь (search_with_profile + rerank_chunks) запускать только если:
  - заполнены >= 2 непустых слота, ЛИБО
  - заполнен «сильный» слот: user_type ИЛИ document_hints.
Один слабый слот (например topics из одного слова) НЕ должен триггерить реранкер.

Реализация: вынести логику в отдельную функцию profile_signal_strength(profile)
или явный предикат; заменить ею текущий bool(... or ... or ...).

Критерии готовности:
- Профиль с одним слабым слотом topics → идёт дешёвый путь _retriever.search.
- Профиль с user_type или document_hints → дорогой путь.
- Профиль с >=2 слотами → дорогой путь.
```

---

# Фаза B. Метрики и dataset

---

## B1 — Строгий JSON-schema output классификатора (§4.4)

```text
Контекст: Telegram-бот AITU. Файлы rag/dialog/classifier.py, rag/generator.py,
config.py.

Задача: убрать класс ошибок «модель завернула JSON в markdown».
1. Проверить, поддерживает ли текущий провайдер (Groq / OpenRouter, см.
   LLM_PROVIDER и llm_model в config.py) параметр
   response_format={"type": "json_schema", "json_schema": {...}}.
2. Если поддерживает — передать в вызов classify_intent строгую json_schema
   для {needs_clarification: bool, reason: enum, confidence: number}.
   Поиск «первого {» оставить как fallback на случай провайдеров без поддержки.
3. Если провайдер НЕ поддерживает — не ломать ничего, оставить текущий парсинг
   и зафиксировать вывод в отчёте.

Сделать это ДО снятия метрик (B4/B6), чтобы мерить уже стабильный парсер.

Критерии готовности:
- При поддержке провайдером — classify_intent использует json_schema.
- Парсинг по-прежнему имеет fallback и не падает.
- В ответе агента указано, поддерживает ли реальный провайдер json_schema.
```

---

## B2 — Feedback на качество уточнений (§4.5)

```text
Контекст: Telegram-бот AITU. Файлы duplicate_detection/db.py,
bot/handlers/feedback.py, bot/handlers/dialog.py.

Задача: собирать сигнал «помогло ли уточнение».
1. Схема: добавить колонку was_clarification_helpful в query_logs
   (идемпотентная миграция, как в A7).
2. Для запросов, прошедших уточняющий диалог (clarification_rounds > 0),
   после выдачи ответа показать дополнительную inline-кнопку обратной связи
   «Уточнение помогло? 👍/👎» — отдельно от обычного thumbs up/down ответа.
3. Обработать callback и записать was_clarification_helpful в строку query_logs
   по log_id.

Через 1–2 недели накопится 200–500 размеченных кейсов для оценки
precision/recall классификатора и A/B промптов.

Критерии готовности:
- Колонка was_clarification_helpful есть в query_logs.
- После уточнённого ответа показывается кнопка оценки уточнения.
- Клик пишет значение в БД; обычная обратная связь по ответу не сломана.
```

---

## B3 — Накопление 200+ реальных запросов `[операционное]`

```text
Контекст: Telegram-бот AITU. Эксплуатация, не код.

Задача: после выката фазы A и B1/B2 оставить бота в работе, пока в таблице
query_logs (data/bot.db) не накопится >= 200 реальных запросов с заполненными
classification_reason (из A7).

Проверка готовности — SQL:
  SELECT COUNT(*) FROM query_logs WHERE classification_reason IS NOT NULL;

Это вход для B4 (распределение score) и B5 (golden set). Двигаться к фазе C
до накопления данных НЕЛЬЗЯ — пороги триажа калибруются на этих данных.
```

---

## B4 — Скрипт снятия распределения score (§3.3, вход для фазы C)

```text
Контекст: Telegram-бот AITU. RAG: эмбеддер intfloat/multilingual-e5-large
(cosine, L2-норм, префиксы query:/passage:), Qdrant. Корпус 70–120 документов.

Задача: написать скрипт scripts/score_distribution.py (создать каталог scripts/),
который:
1. Берёт >= 200 реальных запросов из query_logs (data/bot.db).
2. Для каждого делает probe-поиск: embed(query) → Qdrant search(k=15).
3. Считает по каждому запросу признаки: top1, top2, gap=top1-top2,
   mean(top-5), mean(top-15), std(top-K), doc_spread (число разных
   doc_title в top-5), нормированную энтропию распределения score.
4. Выводит перцентили (p10/p25/p50/p75/p90) по каждому признаку и сохраняет
   per-query таблицу в CSV (scripts/score_distribution.csv).

ВАЖНО (из дизайн-документа): e5-модель даёт сжатое распределение косинуса
в полосе ≈0.70–0.92. Не предлагать пороги «с потолка» — задача скрипта именно
ИЗМЕРИТЬ реальное распределение, чтобы потом взять пороги из перцентилей.

Критерии готовности:
- Скрипт запускается: python -m scripts.score_distribution
- На выходе CSV + печать перцентилей по всем признакам.
- Переиспользует существующие rag/embedder.py и rag/retriever.py
  (не дублировать логику эмбеддинга).
```

---

## B5 — Ручная разметка golden set из 100 кейсов `[операционное + код]`

```text
Контекст: Telegram-бот AITU.

Задача: собрать golden set для регрессионного теста классификатора.
1. Написать скрипт scripts/export_for_labeling.py — выгружает 100 реальных
   запросов из query_logs в файл data/golden_set.jsonl, по одному на строку:
   {"query": "...", "expected_needs_clarification": null, "expected_reason": null}
   (поля expected_* заполняются вручную потом). Стратифицировать выборку:
   примерно поровну запросов с reason=specific и vague_topic/ambiguous.
2. [операционное] Разметить 100 кейсов вручную: проставить
   expected_needs_clarification (true/false) и expected_reason
   (specific / vague_topic / ambiguous).
3. Описать формат data/golden_set.jsonl в README или комментарии скрипта.

Критерии готовности:
- scripts/export_for_labeling.py создаёт data/golden_set.jsonl со 100 строками.
- Структура файла зафиксирована и пригодна для B6.
```

---

## B6 — Скрипт автотеста на golden set (§5 фаза B)

```text
Контекст: Telegram-бот AITU. В проекте нет тестов — это первый.
Зависит от: B5 (data/golden_set.jsonl размечен).

Задача: написать scripts/eval_classifier.py, который:
1. Читает data/golden_set.jsonl.
2. Для каждого кейса вызывает classify_intent(query).
3. Сравнивает с expected_needs_clarification и expected_reason.
4. Печатает метрики: precision, recall, F1 для needs_clarification,
   confusion matrix по reason, список расхождений (query, expected, got).
5. Поддерживает флаг --baseline <file>: сохранить/сравнить прогон, чтобы
   видеть регрессию после правок промпта.

Этот скрипт — регрессионный тест для всех последующих промптов фаз C/E.

Критерии готовности:
- python -m scripts.eval_classifier печатает precision/recall/F1 и расхождения.
- Прогон на текущем классификаторе зафиксирован как baseline
  (scripts/eval_baseline_phaseA.json).
```

---

# Фаза C. Retrieval-grounded triage

---

## C1 — Probe retrieval + правила триажа, Stages 1–4 (§3.2)

```text
Контекст: Telegram-бот AITU, RAG. Эмбеддер intfloat/multilingual-e5-large.
Файлы: новый rag/dialog/triage.py, rag/retriever.py, bot/handlers/user.py,
config.py.
Зависит от: B4 (распределение score снято).

Задача: реализовать каскад классификации намерения ДО LLM-классификатора.
Создать модуль rag/dialog/triage.py со стадиями:

Stage 1 — дешёвые эвристики (без LLM):
  - запрос из 1 слова ИЛИ только из стоп-слов → needs_clarification=True,
    reason="too_short".
  - Порог строго по 1 слову: «length < 3» НЕЛЬЗЯ — двухсловные именные группы
    («академический отпуск», «стоимость пересдачи») наиболее поисковопригодны.

Stage 2 — probe retrieval: embed(query) → Qdrant search(k=15). ВЕКТОР запроса
  вернуть наружу, чтобы переиспользовать его в финальном поиске (Stage 6,
  фаза D) — probe не должен добавлять лишний embed-вызов.

Stage 3 — триаж по score distribution (без LLM), правила A/B/C/D. Числовые
  пороги НЕ хардкодить в коде — брать из config.py (см. C2). Признаки —
  масштабонезависимые (gap как отношение к std, энтропия, перцентильный ранг),
  а не абсолютные косинусы. Если пороги ещё не калиброваны — Stage 3 должен
  всегда отдавать ветку D (передать в LLM), чтобы не сломать поведение.

Stage 4 — LLM verdict только для пограничных (ветка D): в промпт классификатора
  добавить doc_titles + section_titles из top-5 probe-результата.

Включить весь каскад через feature-flag в config.py (TRIAGE_ENABLED, default
False) и в SHADOW-режиме: логировать вердикт триажа РЯДОМ со старым
classify_intent, поведение хендлера пока не менять. Сравнение результатов —
в логи/query_logs.

Критерии готовности:
- rag/dialog/triage.py реализует Stages 1–4, probe-вектор возвращается наружу.
- При TRIAGE_ENABLED=False поведение бота не меняется (только shadow-логи).
- Пороги читаются из config.py, не захардкожены.
```

---

## C2 — Калибровка порогов триажа и вынос в config (§3.3, §3.5 R2)

```text
Контекст: Telegram-бот AITU. Файлы config.py, scripts/score_distribution.py
(из B4), rag/dialog/triage.py.
Зависит от: B4, C1.

Задача: задать пороги триажа из РЕАЛЬНОГО распределения, а не round-чисел.
1. По CSV из B4 (scripts/score_distribution.csv) взять пороги из перцентилей:
   - out_of_scope — согласовать с уже существующим в config.py
     min_chunk_score = 0.55 («шумовой пол»). НЕ вводить параллельную
     константу 0.50 — опираться на min_chunk_score.
   - high-confidence top1, gap, top-K mean, doc_spread — из перцентилей.
2. Добавить эти пороги в config.py (Settings) с осмысленными именами и
   комментариями, что они калиброваны на распределении от <дата>.
3. Написать scripts/recalibrate_triage.py — регенерация порогов из свежего
   score_distribution.csv (для R2: пороги дрейфуют при смене эмбеддера/корпуса).
4. triage.py должен читать пороги только из config.py.

ВАЖНО: иллюстративные числа из §3.3 (0.80 / 0.15 / 0.70 / 0.50) НЕ использовать
как дефолты — на e5 правила A и B с ними не срабатывают никогда. Только
перцентили реальных данных.

Критерии готовности:
- Пороги триажа в config.py откалиброваны по B4-данным.
- out_of_scope-порог выражен через min_chunk_score.
- scripts/recalibrate_triage.py пересчитывает пороги из CSV.
```

---

## C3 — Верификация top1 высокой confidence, R1 (§3.5)

```text
Контекст: Telegram-бот AITU. Файлы rag/dialog/triage.py, rag/generator.py.
Зависит от: C1, C2.

Задача: закрыть риск ложного «прямого ответа» по правилу B триажа.
Cosine similarity 0.85 != «семантически тот же вопрос» — высокий top1 ещё
не гарантирует, что чанк реально отвечает на запрос.

Mitigation (выбрать дешёвый вариант):
- Вариант 1: при срабатывании правила B (specific, прямой ответ) сделать
  ОДИН cheap LLM-reranker-вызов ТОЛЬКО по этому одному top1-чанку — подтвердить
  релевантность. Не подтвердилось → откатить вердикт в ветку D (LLM verdict).
- Вариант 2: положиться на этап генерации с явным «I don't know»-промптом,
  если чанк не отвечает на вопрос.

Реализовать вариант 1 (он строже и дешевле полного реранка).

Критерии готовности:
- Правило B триажа подтверждается одним LLM-вызовом по top1-чанку.
- При неподтверждении вердикт уходит в ветку D, а не сразу в прямой ответ.
```

---

## C4 — Явный raise вместо тихого fallback в классификаторе (§4.8)

```text
Контекст: Telegram-бот AITU. Файл rag/dialog/classifier.py.
Зависит от: C1 (нужен «fallback на правилах» из Stage 1/3).

Задача: закрыть тихую дыру в classify_intent. Разобраться в ТРЁХ строках:

- Строка 48: needs = bool(data.get("needs_clarification")) — НАСТОЯЩАЯ дыра.
  Если ключа нет, bool(None)=False, и ответ бесшумно становится «уточнение
  не нужно». ИСПРАВИТЬ: отсутствие ключа needs_clarification должно явно
  падать в ValueError. В except — fallback на правила триажа из rag/dialog/
  triage.py (Stage 1/3), а не молчаливый needs_clarification=False.

- Строка 51: reason = "vague_topic" if needs else "specific" — это НЕ дыра.
  JSON уже распарсен, needs_clarification прочитан; строка срабатывает только
  при reason вне валидного множества. Вывод reason из needs здесь корректен —
  НЕ ТРОГАТЬ.

- Строка 53: except уже корректно ловит реальные ошибки парсинга и возвращает
  reason="error" — поведение менять НЕ нужно (кроме замены fallback-вердикта
  на правила, см. выше).

Критерии готовности:
- Отсутствие ключа needs_clarification → ValueError, затем fallback на
  правила триажа.
- Строки 51 и логика except-reason="error" сохранены.
- Прогнать scripts/eval_classifier.py (B6) — регрессии относительно baseline нет.
```

---

## C5 — Прогон фазы C на golden set и сравнение (§5 фаза C)

```text
Контекст: Telegram-бот AITU.
Зависит от: B6, C1–C4.

Задача: оценить триаж против baseline фазы A.
1. Расширить scripts/eval_classifier.py: гонять не только classify_intent,
   но и полный каскад триажа (rag/dialog/triage.py).
2. Прогнать на data/golden_set.jsonl, сравнить метрики с
   scripts/eval_baseline_phaseA.json.
3. Сохранить отчёт scripts/eval_phaseC_report.md: precision/recall/F1 до и
   после, разбор регрессий, рекомендация — включать TRIAGE_ENABLED=True или нет.

Критерии готовности:
- eval_classifier.py умеет оценивать каскад триажа.
- Есть отчёт со сравнением фаза A vs фаза C.
- Дано явное решение по feature-flag TRIAGE_ENABLED.
```

---

# Фаза D. Grounded clarification

---

## D1 — Передача top-15 в `next_clarification`, Stages 5–6 (§3.2)

```text
Контекст: Telegram-бот AITU. Файлы rag/dialog/question_gen.py,
bot/handlers/dialog.py, rag/dialog/triage.py.
Зависит от: C1 (probe retrieval даёт top-15 и его doc-список).

Задача: уточняющие вопросы должны опираться на реально найденные документы,
а не на весь корпус.
1. next_clarification (rag/dialog/question_gen.py) сейчас получает полный
   список документов корпуса. Изменить так, чтобы он получал top-15 чанков
   probe-поиска (с doc_title, section_title) — варианты ответов LLM выбирает
   ТОЛЬКО из этих документов.
2. Протащить top-15 из триажа через FSM-состояние: сохранить в
   start_clarification_dialog (bot/handlers/dialog.py), читать в _ask_next /
   _generate_question.
3. _generate_question больше не вызывает _get_cached_docs(_retriever) для
   полного корпуса — использует сохранённый top-15.

Критерии готовности:
- next_clarification генерирует варианты только из документов top-15.
- top-15 хранится в FSM и доступен на всех раундах диалога.
- Уточняющий вопрос для «Расскажи про правила» предлагает конкретные
  найденные документы, а не абстрактные оси.
```

---

## D2 — Фильтрация чанков по ответам без повторного embed, Stage 6–7 (§3.2)

```text
Контекст: Telegram-бот AITU. Файлы bot/handlers/dialog.py, rag/dialog/triage.py.
Зависит от: C1 (probe-вектор переиспользуется), D1.

Задача: после ответов пользователя фильтровать уже имеющийся top-15, а не
делать новый поиск с нуля.
В _proceed_to_search (bot/handlers/dialog.py):
1. Отфильтровать сохранённый top-15 по собранным ответам: doc_title,
   section_title, user_type keyword, document_hints из профиля.
2. Если после фильтра осталось >= settings.top_k чанков — идти сразу в
   генерацию, БЕЗ повторного embed и поиска.
3. Если осталось < top_k — дополнить новым поиском по enriched query.
   Использовать ПЕРЕИСПОЛЬЗОВАННЫЙ probe-вектор, где это возможно, чтобы не
   делать лишний embed-вызов.

Цель — путь «1 раунд диалога → ответ» без повторного эмбеддинга
(см. пример §6: «Общежитие» → filter_chunks → generate, embed не нужен).

Критерии готовности:
- При достаточном числа чанков после фильтра повторного поиска нет.
- При нехватке чанков добавляется поиск, по возможности без нового embed.
- Уточнённый ответ собирается из подтверждённых документов.
```

---

## D3 — Предсчёт кратких описаний документов (§4.7)

```text
Контекст: Telegram-бот AITU. Корпус 70–120 документов.
Файлы: ingestion/ingest.py, rag/dialog/question_gen.py, duplicate_detection/db.py
(или новая таблица), config.py.

Задача: предкэшировать краткое описание (1–2 предложения) каждого документа
и использовать его в next_clarification для более осмысленных вариантов.
1. При индексации документа (ingestion/ingest.py) один раз сгенерировать
   LLM-саммари документа (1–2 предложения) из его чанков.
2. Хранить саммари: либо в payload Qdrant, либо в отдельной SQLite-таблице
   doc_summaries (data/bot.db). Выбрать SQLite-таблицу — проще
   инвалидировать/перечитывать; миграция идемпотентная (как в A7).
3. next_clarification передавать саммари top-15 документов на вход LLM —
   варианты выбора станут осмысленнее.
4. При удалении документа (/delete) удалять и его саммари.

Критерии готовности:
- При индексации документа создаётся и сохраняется его краткое саммари.
- next_clarification использует саммари как контекст.
- Удаление документа чистит саммари.
```

---

# Фаза E. Контекст и polish (опционально)

---

## E1 — Conversation-aware classification, follow-up detection (§4.1)

```text
Контекст: Telegram-бот AITU. Файлы bot/handlers/user.py, rag/dialog/classifier.py
(или новый rag/dialog/followup.py), rag/dialog/prompts.py.

Задача: распознавать follow-up вопросы, чтобы не запускать диалог заново.
Кейс: после завершённого уточнения follow-up «А для магистрантов?» снова
классифицируется как vague_topic и снова запускает диалог.

1. Хранить последний завершённый original_query и profile пользователя
   с коротким TTL (~5 мин) — в FSM или in-memory dict по user_id.
2. При новом сообщении, если есть свежий контекст, сделать LLM-вызов:
   дан [last_turn_context] + [new_message] — это follow-up предыдущего
   вопроса (использовать тот же профиль) или новый вопрос (классифицировать
   заново)?
3. Если follow-up — переиспользовать сохранённый profile, слить с новым
   сообщением и идти сразу в поиск, минуя уточняющий диалог.

Критерии готовности:
- После уточнённого диалога «А для магистрантов?» в течение 5 мин
  распознаётся как follow-up и не запускает новый диалог.
- Просроченный (> 5 мин) или несвязанный запрос классифицируется заново.
```

---

## E2 — Hybrid retrieval: BM25 + dense fusion (§4.2)

```text
Контекст: Telegram-бот AITU, Qdrant. Файлы rag/retriever.py, ingestion/ingest.py,
config.py.

Задача: добавить sparse (BM25) + dense гибридный поиск как альтернативу
переуточнению. Keyword-задачи (например «правила») embedding-модель решает
хуже BM25.
1. Включить sparse-вектора в схему коллекции Qdrant (Qdrant поддерживает
   sparse + dense hybrid).
2. При индексации (ingestion/ingest.py) считать и upsert-ить sparse-вектора
   чанков вместе с dense.
3. В rag/retriever.py добавить hybrid-поиск с fusion (RRF или score fusion)
   dense + sparse результатов.
4. Под feature-flag в config.py (HYBRID_SEARCH_ENABLED).

Ожидаемый эффект на корпусе 70–120 документов: +5–10% recall@5.
ВНИМАНИЕ: потребуется переиндексация корпуса — предусмотреть это в инструкции.

Критерии готовности:
- Коллекция Qdrant хранит sparse + dense вектора.
- retriever.py делает hybrid-поиск с fusion под feature-flag.
- Документирована необходимость переиндексации.
```

---

## E3 — Кэш классификатора (§4.3)

```text
Контекст: Telegram-бот AITU. Файл rag/dialog/classifier.py.

Задача: кэшировать результат классификатора по hash вопроса на 24 часа —
на FAQ-подобном трафике дубликаты неизбежны.
1. Нормализовать вопрос (lower, trim, схлопнуть пробелы) → ключ кэша.
2. In-memory кэш {hash: (ClassificationResult, expires_at)} с TTL 24 ч.
3. classify_intent сначала смотрит в кэш, при miss — LLM-вызов и запись.

Замечание из дизайн-документа: ценность этого кэша ПАДАЕТ после фазы C —
триаж и так делает быстрый путь без LLM. Делать только если фаза C не
покрыла основной трафик. Низкий приоритет.

Критерии готовности:
- Повторный идентичный запрос в пределах 24 ч не вызывает LLM.
- Кэш не растёт безгранично (ограничение размера или периодическая чистка).
```

---

## E4 — Адаптивное число раундов диалога (§4.6)

```text
Контекст: Telegram-бот AITU. Файл bot/handlers/dialog.py.

Задача: сейчас жёсткий лимит _MAX_ROUNDS = 3 (bot/handlers/dialog.py:31).
Сделать число раундов адаптивным.

В _ask_next (bot/handlers/dialog.py) после получения ответа пользователя:
если профиль уже «сильный» (>= 2 непустых слота — переиспользовать предикат
силы профиля из A8), завершить диалог досрочно и идти в _proceed_to_search,
не задавая оставшиеся вопросы.

Жёсткий потолок 3 раунда оставить как верхнюю границу.

Критерии готовности:
- Если после 1-го ответа профиль сильный (>=2 слота) — диалог завершается
  одним раундом.
- Слабый профиль по-прежнему может дойти до 3 раундов.
- Используется общий предикат силы профиля из A8 (без дублирования логики).
```

---

# Чек-лист выполнения

Фаза A — Quick wins
- [ ] A1 — max_tokens=50
- [ ] A2 — confidence в выходе классификатора
- [ ] A3 — KZ stop-words
- [ ] A4 — инвалидация doc-кэша при upload
- [ ] A5 — объединить enrich + profile
- [ ] A6 — доменные few-shot примеры
- [ ] A7 — лог classification_reason в БД
- [ ] A8 — порог силы профиля для дорогого пути

Фаза B — Метрики и dataset
- [ ] B1 — строгий JSON-schema output
- [ ] B2 — feedback на качество уточнений
- [ ] B3 — накопить 200+ запросов `[операционное]`
- [ ] B4 — скрипт распределения score
- [ ] B5 — golden set на 100 кейсов
- [ ] B6 — скрипт автотеста на golden set

Фаза C — Retrieval-grounded triage
- [ ] C1 — probe retrieval + триаж, Stages 1–4
- [ ] C2 — калибровка порогов + вынос в config
- [ ] C3 — верификация top1 (R1)
- [ ] C4 — явный raise в классификаторе
- [ ] C5 — прогон на golden set, сравнение

Фаза D — Grounded clarification
- [ ] D1 — top-15 в next_clarification
- [ ] D2 — фильтрация чанков без повторного embed
- [ ] D3 — предсчёт doc summaries

Фаза E — Контекст и polish (опционально)
- [ ] E1 — follow-up detection
- [ ] E2 — hybrid search
- [ ] E3 — кэш классификатора
- [ ] E4 — адаптивное число раундов
