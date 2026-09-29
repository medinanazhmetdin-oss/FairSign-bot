import os
import re
import io
import threading

import telebot
from flask import Flask
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =========================
# НАСТРОЙКИ
# =========================

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Не задана переменная TELEGRAM_TOKEN (проверь .env)")
if not GEMINI_API_KEY:
    raise RuntimeError("Не задана переменная GEMINI_API_KEY (проверь .env)")
GEMINI_MODEL = "gemini-3.5-flash-lite"

MAX_CONTRACT_CHARS = 15000

# =========================
# ПОДКЛЮЧЕНИЕ
# =========================

bot = telebot.TeleBot(TELEGRAM_TOKEN)
ai = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# ФИЛЬТР МАТА
# =========================

_PROFANITY_ROOTS = [
    r"бля[дт]",
    r"хуй|хуе|хуё",
    r"пизд",
    r"еба[тн]|ёба[тн]|ебал|ёбан",
    r"муда[кч]",
    r"сука|суч[ае]",
    r"гандон",
    r"долбоеб|долбоёб",
]
_PROFANITY_PATTERN = re.compile(
    r"\b(" + "|".join(_PROFANITY_ROOTS) + r")[а-яё]*\b",
    re.IGNORECASE,
)


def censor_profanity(text: str):
    found = False

    def _mask(match):
        nonlocal found
        found = True
        word = match.group(0)
        return word[0] + "*" * (len(word) - 1)

    new_text = _PROFANITY_PATTERN.sub(_mask, text)
    return new_text, found


# =========================
# ПРИВЕТСТВИЯ (не блокируем)
# =========================

_GREETING_PATTERN = re.compile(
    r"^\s*(привет|здравствуй|добрый день|добрый вечер|доброе утро|hi|hello)\W*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    return bool(_GREETING_PATTERN.match(text))


# =========================
# ЧИСТКА ТЕКСТА ОТ "ИИШНОГО" ОФОРМЛЕНИЯ
# =========================

def humanize(text: str) -> str:
    """Убираем markdown-мусор, который Gemini любит добавлять
    (звёздочки, решётки, обратные кавычки) — в Телеграме без parse_mode
    он выглядит как кракозябры и сразу выдаёт нейросеть."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # **жирный**
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)\*(?!\w)", r"\1", text)  # *курсив*
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # ### заголовки
    text = re.sub(r"^\s*[\*\u2022]\s+", "— ", text, flags=re.MULTILINE)  # * пункты -> —
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)            # лишние пустые строки
    return text.strip()


# =========================
# ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ ФАЙЛОВ
# =========================

def extract_text_from_docx(file_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


# =========================
# ПРОМПТЫ
# =========================

ANALYSIS_SYSTEM_PROMPT = """\
Ты — Qarasozshe, практикующий юрист по трудовому праву Республики Казахстан
с многолетним опытом. К тебе обратился обычный человек с трудовым договором
и хочет понять, что он подписывает. Разбери договор по Трудовому кодексу РК
(ТК РК) точно и по делу, как будто клиент сидит напротив тебя в кабинете.

КАК ПИСАТЬ (это самое важное правило, следуй ему строго):
- Пиши живым человеческим языком, на «вы», как юрист говорит с клиентом
  на консультации: спокойно, уверенно, без канцелярита.
- Никогда не упоминай, что ты ИИ, нейросеть, модель, бот или программа.
  Не извиняйся за "ограничения модели". Ты просто юрист, который смотрит
  документ.
- Никакого markdown: без звёздочек, решёток, подчёркиваний, обратных
  кавычек, таблиц. Только обычный текст и абзацы, как в письме.
- Не используй нумерованные разделы с заглавными заголовками вроде
  "1. ОБЩИЕ СВЕДЕНИЯ". Пиши связным текстом, короткими абзацами, будто
  наговариваешь клиенту вывод. Если нужен список — через тире «—».
- Эмодзи не используй вообще, ни одного. Никаких цветных кружков,
  значков предупреждения, галочек.
- Критичность передавай словами: «это серьёзно, я бы не подписывал, пока
  не исправят», «стоит поправить, но не смертельно», «мелочь, но лучше
  учесть».
- Строго избегай фраз, которые выдают нейросеть: «Безусловно», «Важно
  отметить», «Стоит подчеркнуть», «В заключение», «Данный документ»,
  «Подводя итог», «Как я уже сказал», «В целом можно сказать». Пиши так,
  как реальный человек формулирует мысль сходу, а не по шаблону.
- Не начинай с длинного вступления и не пересказывай, что тебя попросили
  сделать. Сразу переходи к сути, как будто продолжаешь разговор.
- Иногда уместна лёгкая разговорность юриста: «смотрите, тут вот что»,
  «здесь есть нюанс», «это классическая уловка работодателя» — но без
  перегиба и не в каждом абзаце.
- Не хвали договор, не благодари за вопрос, не желай удачи в конце.
- Лучше короче и по делу, чем длинно и обтекаемо.

ЧТО НУЖНО СКАЗАТЬ (порядок примерно такой, без формальных заголовков):
1. Коротко, что это за договор: срочный или бессрочный, кто стороны, срок,
   должность, зарплата, режим работы, испытательный срок. Указывай только
   то, что реально есть в тексте. Чего нет — так и скажи: «в договоре это
   не прописано».
2. Что настораживает. По каждой проблеме — отдельный абзац: что именно
   написано в договоре (коротко процитируй формулировку), чем это плохо
   и на какую статью ТК РК это опирается (например: ст. 23 — дискриминация,
   ст. 28 — обязательные условия договора, ст. 36 — испытательный срок,
   ст. 65 — рабочее время, ст. 71-72 — сверхурочная работа, ст. 88 —
   отпуска, ст. 131-132 — расторжение договора, ст. 113 — материальная
   ответственность), насколько это серьёзно и что конкретно попросить
   изменить. Если существенных проблем нет — прямо скажи об этом, не
   выдумывай риски ради количества.
3. Чего не хватает из обязательного по ст. 28 ТК РК, и какую формулировку
   стоит попросить добавить.
4. Итог: можно ли подписывать как есть, что обязательно поправить до
   подписи, и общий уровень риска (низкий, средний, высокий).

ТОЧНОСТЬ (это не обсуждается):
- Не приписывай договору того, чего в нём нет.
- Если текст обрезан или неполный — скажи об этом в итоге.
- Если это вообще не трудовой договор — так и скажи, без выдуманного
  анализа.
- В самом конце одной короткой фразой, без пафоса, напомни, что это
  предварительный разбор и он не заменяет очную консультацию юриста.
"""

CHAT_SYSTEM_PROMPT = """\
Ты — Qarasozshe, практикующий юрист, который помогает разбираться в
трудовых договорах по законодательству Республики Казахстан. Отвечай
коротко (2-4 предложения), как юрист отвечает клиенту в переписке: на
«вы», по-человечески, без канцелярита.

Никогда не упоминай, что ты ИИ, нейросеть, модель или бот. Без markdown,
без звёздочек и решёток, без списков и без единого эмодзи. Не используй
шаблонные фразы вроде «Безусловно», «Важно отметить», «В целом».

Если вопрос про трудовое право РК — ответь по существу, коротко и уверенно.
Если вопрос не по теме — вежливо, без занудства, скажи, что вы разбираете
трудовые договоры, и предложите прислать текст или файл договора.
"""


# =========================
# ВЫЗОВЫ GEMINI
# =========================

def analyze_contract(contract_text: str) -> str:
    contract_text = contract_text[:MAX_CONTRACT_CHARS]

    response = ai.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Вот текст трудового договора:\n\n{contract_text}",
        config=types.GenerateContentConfig(
            system_instruction=ANALYSIS_SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=3500,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini вернул пустой ответ")
    return humanize(response.text)


def quick_answer(user_text: str) -> str:
    response = ai.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM_PROMPT,
            temperature=0.5,
            max_output_tokens=500,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini вернул пустой ответ")
    return humanize(response.text)


def send_long_message(chat_id, text: str):
    """Режем по абзацам, а не посреди слова."""
    limit = 4000
    chunk = ""
    for paragraph in text.split("\n\n"):
        if len(chunk) + len(paragraph) + 2 > limit and chunk:
            bot.send_message(chat_id, chunk.strip())
            chunk = ""
        # на случай абзаца длиннее лимита
        while len(paragraph) > limit:
            bot.send_message(chat_id, paragraph[:limit])
            paragraph = paragraph[limit:]
        chunk += paragraph + "\n\n"
    if chunk.strip():
        bot.send_message(chat_id, chunk.strip())


ERROR_TEXT = (
    "Что-то пошло не так, не получилось сделать разбор. "
    "Попробуйте ещё раз через минутку."
)


# =========================
# START
# =========================

@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "Привет! Я Qarasozshe. Помогаю разобраться в трудовом договоре по законам "
        "Казахстана, чтобы вы не подписали лишнего.\n\n"
        "Просто пришлите текст договора или файл (.docx, .pdf, .txt), и я скажу, "
        "что в нём не так, чего не хватает и стоит ли подписывать.\n\n"
        "Если в сообщении будет мат, я его замаскирую."
    )


# =========================
# ФАЙЛЫ (.docx / .pdf / .txt)
# =========================

@bot.message_handler(content_types=["document"])
def handle_document(message):
    caption = message.caption or ""
    if caption:
        clean_caption, had_profanity = censor_profanity(caption)
        if had_profanity:
            bot.send_message(
                message.chat.id,
                f"В подписи к файлу был мат, я его скрыл:\n\n{clean_caption}",
            )

    file_name = (message.document.file_name or "").lower()
    file_info = bot.get_file(message.document.file_id)
    file_bytes = bot.download_file(file_info.file_path)

    try:
        if file_name.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        elif file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif file_name.endswith(".txt"):
            text = file_bytes.decode("utf-8", errors="ignore")
        else:
            bot.send_message(message.chat.id, "Я умею читать только .docx, .pdf и .txt.")
            return
    except Exception as error:
        print("Ошибка чтения файла:", error)
        bot.send_message(
            message.chat.id,
            "Не получилось открыть этот файл. Попробуйте прислать другой или вставьте текст договора сообщением.",
        )
        return

    if len(text.strip()) < 100:
        bot.send_message(
            message.chat.id,
            "В файле почти нет текста. Похоже, это скан или фото. "
            "Пришлите версию с текстом или вставьте текст сообщением.",
        )
        return

    clean_text, _ = censor_profanity(text)

    bot.send_chat_action(message.chat.id, "typing")
    bot.send_message(message.chat.id, "Читаю договор, минутку...")

    try:
        result = analyze_contract(clean_text)
    except Exception as error:
        print("Ошибка Gemini:", error)
        bot.send_message(message.chat.id, ERROR_TEXT)
        return

    send_long_message(message.chat.id, result)


# =========================
# ТЕКСТОВЫЕ СООБЩЕНИЯ
# =========================

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    raw_text = message.text or ""

    if is_greeting(raw_text):
        bot.send_message(
            message.chat.id,
            "Здравствуйте! Пришлите текст трудового договора или файл, и я его разберу.",
        )
        return

    clean_text, had_profanity = censor_profanity(raw_text)
    if had_profanity:
        bot.send_message(
            message.chat.id,
            f"Мат я скрыл:\n\n{clean_text}",
        )

    if len(clean_text.strip()) < 100:
        try:
            bot.send_chat_action(message.chat.id, "typing")
            bot.send_message(message.chat.id, quick_answer(clean_text))
        except Exception as error:
            print("Ошибка Gemini:", error)
            bot.send_message(message.chat.id, ERROR_TEXT)
        return

    bot.send_chat_action(message.chat.id, "typing")
    bot.send_message(message.chat.id, "Читаю договор, минутку...")

    try:
        result = analyze_contract(clean_text)
    except Exception as error:
        print("Ошибка Gemini:", error)
        bot.send_message(message.chat.id, ERROR_TEXT)
        return

    send_long_message(message.chat.id, result)


# =========================
# KEEP-ALIVE СЕРВЕР (для Render — обязателен)
# =========================

keep_alive_app = Flask(__name__)


@keep_alive_app.route("/")
def home():
    return "Qarasozshe bot is running!"


def run_keep_alive():
    port = int(os.environ.get("PORT", 3000))
    keep_alive_app.run(host="0.0.0.0", port=port)


def start_keep_alive():
    t = threading.Thread(target=run_keep_alive)
    t.daemon = True
    t.start()


# =========================
# ЗАПУСК
# =========================

start_keep_alive()
print("Бот запущен!")
bot.infinity_polling()
