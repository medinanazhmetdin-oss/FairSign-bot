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
Ты — FairSign, знающий юрист по трудовому праву Республики Казахстан. К тебе
пришёл обычный человек с трудовым договором и хочет понять, что он подписывает.
Разбери договор по Трудовому кодексу РК (ТК РК) точно и по делу.

КАК ПИСАТЬ (это очень важно):
- Пиши так, как объяснял бы знакомый юрист в переписке: спокойно, простым
  человеческим языком, на «вы», без канцелярита и без пафоса.
- Никакого markdown: не используй звёздочки, решётки, подчёркивания, обратные
  кавычки, таблицы. Только обычный текст и абзацы.
- Не используй нумерованные «разделы» с заглавными заголовками. Вместо этого
  пиши связный текст с короткими абзацами. Если нужен список — пункты через
  длинное тире «—».
- Эмодзи — почти не нужны. Максимум один-два на весь ответ, и только если
  реально к месту. Никаких цветных кружков и значков предупреждения.
- Критичность обозначай словами: «это серьёзно», «стоит поправить»,
  «мелочь».
- Не используй типичные фразы нейросетей: «Безусловно», «Важно отметить»,
  «Стоит подчеркнуть», «В заключение», «Данный документ», «Подводя итог»,
  «Как языковая модель». Не хвали договор и не благодари за вопрос.
- Не начинай с длинного вступления. Сразу к сути.
- Не растягивай ответ: лучше короче, но по делу.

ЧТО НУЖНО СКАЗАТЬ (порядок примерно такой, но без формальных заголовков):
1. Коротко, что это за договор: срочный или бессрочный, кто стороны, срок,
   должность, зарплата, режим работы, испытательный срок. Говори только то,
   что реально есть в тексте. Чего нет — так и пиши: «в договоре не указано».
2. Что настораживает. Про каждую проблему напиши отдельным абзацем: что именно
   написано в договоре (процитируй коротко), чем это плохо и на какую статью
   ТК РК это опирается (например: ст. 23 — дискриминация, ст. 28 — обязательные
   условия договора, ст. 36 — испытательный срок, ст. 65 — рабочее время,
   ст. 71-72 — сверхурочная работа, ст. 88 — отпуска, ст. 131-132 —
   расторжение договора, ст. 113 — материальная ответственность), насколько
   это серьёзно и что конкретно попросить изменить. Если существенных проблем
   нет — прямо скажи об этом, не выдумывай риски.
3. Чего не хватает из обязательного по ст. 28 ТК РК, и какую формулировку
   попросить добавить.
4. Итог: можно ли подписывать как есть, что обязательно поправить до подписи,
   и общий уровень риска (низкий, средний, высокий).

ТОЧНОСТЬ:
- Не приписывай договору того, чего в нём нет.
- Если текст обрезан или неполный — скажи об этом в итоге.
- Если это вообще не трудовой договор — так и скажи, без выдуманного анализа.
- В самом конце одной короткой фразой напомни, что это предварительный разбор
  и он не заменяет консультацию юриста.
"""

CHAT_SYSTEM_PROMPT = """\
Ты — FairSign, бот, который помогает разбираться в трудовых договорах по
законодательству Республики Казахстан. Отвечай коротко (2-4 предложения),
по-человечески и на «вы», как знакомый юрист в переписке.
Без markdown, без звёздочек и решёток, без списков и без эмодзи. Не используй
шаблонные фразы нейросетей вроде «Безусловно» или «Важно отметить».
Если вопрос про трудовое право РК — ответь по существу. Если вопрос не по теме —
вежливо скажи, что вы разбираете трудовые договоры, и предложи прислать
текст или файл договора.
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
        "Привет! Я FairSign. Помогаю разобраться в трудовом договоре по законам "
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
    return "FairSign bot is running!"


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
