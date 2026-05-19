Промпт 1 — Утилита конвертации

В проекте  реализуй утилиту конвертации .docx/.doc файлов в PDF через LibreOffice headless.

Создай файл ingestion/convert.py со следующим:

1. Async функция `convert_to_pdf(src: Path, dest_dir: Path) -> Path`:
   - Запускает LibreOffice: `soffice --headless --convert-to pdf --outdir <dest_dir> <src>`
   - На Windows ищет soffice по пути "C:\Program Files\LibreOffice\program\soffice.exe", 
     на Linux — через shutil.which("soffice")
   - Если LibreOffice не найден — выбрасывает RuntimeError с понятным сообщением
   - Возвращает Path к сконвертированному .pdf файлу
   - Использует asyncio.create_subprocess_exec (не shell=True)
   - Таймаут 60 секунд

Никаких лишних абстракций. Только эта функция.





Промпт 2 — Расширение admin handler

В файле bot/handlers/admin.py расширь команду /upload для поддержки .docx и .doc файлов.

Текущее состояние:
- _PDFS_DIR = Path("pdfs")  (строка 28)
- Валидация на строке 99 принимает только .pdf
- ingest_pdf вызывается на строке 128

Что нужно изменить:

1. Импортируй в начале файла: `from ingestion.convert import convert_to_pdf`

2. Замени валидацию на строке 99:
   - Принимать .pdf, .docx, .doc (case-insensitive)
   - Сообщение об ошибке: "❌ Принимаются только PDF, DOCX и DOC файлы."

3. В функции _run_ingest() перед вызовом ingest_pdf добавь:
   - Если файл .docx или .doc — вызови `await convert_to_pdf(dest, _PDFS_DIR)`
   - Результат (Path к .pdf) используй вместо dest для ingest_pdf и дальше
   - Исходный .docx/.doc файл после конвертации удали (dest.unlink())
   - Обнови переменные doc.file_name → имя pdf файла, title → stem pdf файла
   - Если конвертация падает — отправь сообщение об ошибке и верни return

4. Сообщения статуса:
   - После скачивания .docx/.doc: "⏳ Конвертирую в PDF..."
   - После успешной конвертации: "⏳ Начинаю индексацию в фоне..."

Не меняй ничего кроме описанного.








Промпт 3 — Проверка и тест

В проекте проверь корректность реализации загрузки .docx/.doc для админа:

1. Прочитай ingestion/convert.py и убедись:
   - Функция convert_to_pdf корректно находит soffice на Windows и Linux
   - Используется asyncio.create_subprocess_exec (не subprocess, не shell=True)
   - Есть таймаут и обработка ошибок

2. Прочитай bot/handlers/admin.py и убедись:
   - Валидация принимает .pdf, .docx, .doc
   - После конвертации используется pdf-путь, а не оригинальный .docx/.doc
   - Исходный .docx/.doc файл удаляется после конвертации
   - ingest_pdf получает только .pdf файл
   - Переменные title и имя файла корректно обновлены до .pdf версии

3. Если найдёшь проблемы — исправь их.
4. Напиши в конце что именно проверил и всё ли в порядке.
