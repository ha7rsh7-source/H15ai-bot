import os
import asyncio
import base64
import subprocess
import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from supabase import create_client


# =========================
# CONFIG
# =========================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

OWNER_ID = 1565428409
OWNER_USERNAME = "@Harshupadhyay_15"

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=180.0,
)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY
)

app = FastAPI()

telegram_app = None


# =========================
# STATS
# =========================

stats = {
    "total_messages": 0,
    "total_replies": 0,
    "total_users": 0,
    "total_photos": 0,
    "total_videos": 0,
}

known_users = set()

runtime_errors = 0
last_error = None
last_success = None

photo_tested = False
photo_ok = False

video_tested = False
video_ok = False


def load_stats():
    global stats

    try:
        result = (
            supabase
            .table("bot_stats")
            .select("*")
            .eq("id", 1)
            .execute()
        )

        if result.data:
            row = result.data[0]

            for key in stats:
                stats[key] = int(row.get(key, 0))

    except Exception as e:
        print("Stats load error:", e)


def save_stats():
    try:
        supabase.table("bot_stats").upsert({
            "id": 1,
            **stats
        }).execute()

    except Exception as e:
        print("Stats save error:", e)


def register_user(user_id):
    if user_id not in known_users:
        known_users.add(user_id)
        stats["total_users"] += 1
        save_stats()


def record_error(error):
    global runtime_errors, last_error

    runtime_errors += 1
    last_error = str(error)

    print("ERROR:", error)


def record_success():
    global last_success

    last_success = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


# =========================
# TELEGRAM MESSAGE SPLITTER
# =========================

async def send_long_message(update, text):
    """
    Telegram messages have a character limit.
    Split long AI responses into safe chunks.
    """

    if not text:
        return

    # Safe limit below Telegram's maximum
    limit = 4000

    chunks = []

    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)

        if split_at < 1000:
            split_at = text.rfind(" ", 0, limit)

        if split_at < 1000:
            split_at = limit

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    if text:
        chunks.append(text)

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk)
            await asyncio.sleep(0.1)

        except Exception as e:
            record_error(e)


# =========================
# OWNER CHECK
# =========================

def is_owner(update):

    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are H15ai, Harsh's AI assistant.

Personality:
- Friendly
- Casual
- Hinglish when appropriate
- Talk like a helpful friend
- Can use light memes/emojis naturally
- Don't be unnecessarily formal

Important:
- Accuracy is more important than sounding confident.
- For maths, physics, chemistry and other STEM questions, carefully verify calculations and reasoning before answering.
- If unsure, clearly say so.
- Do not invent facts.
- Give clear explanations.
- Match the user's language.
- Keep simple questions concise.
- For difficult questions, explain step-by-step.

You are NOT Harsh himself.
You are H15ai, his AI bot.
"""


# =========================
# TEXT AI
# =========================

async def text_ai(user_text):

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_text,
            }
        ],
    )

    return response.choices[0].message.content


# =========================
# PHOTO AI
# =========================

async def vision_ai(image_bytes, prompt):

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=950,
        reasoning_effort="none",
    )

    return response.choices[0].message.content


# =========================
# START
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    register_user(update.effective_user.id)

    text = """
🧠 Welcome to H15ai!

I'm Harsh's AI assistant 🤖

You can:
💬 Chat normally
📚 Ask study questions
🧮 Solve maths/physics/chemistry
🖼️ Send photos
🎬 Send videos
😂 Ask for jokes
💡 Get ideas/scripts
💻 Get coding help

Commands:

/start — Start
/help — Help
/clear — Clear your chat context
/about — About H15ai
/stats — Owner health + stats

Just send me anything 😎
"""

    await update.message.reply_text(text)


# =========================
# HELP
# =========================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        """
🧠 H15ai Commands

/start — Start
/help — Help
/clear — Clear your chat context
/about — About H15ai
/stats — Owner health + stats (owner only)

You can also send normal messages, photos and videos.
"""
    )


# =========================
# ABOUT
# =========================

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        """
🤖 H15ai

Harsh's personal AI bot.

Powered by:
🧠 Groq AI
👁️ Vision AI
🗄️ Supabase
✈️ Telegram

Built to be a friendly, useful AI assistant.
"""
    )


# =========================
# CLEAR
# =========================

async def clear_context(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data.clear()

    await update.message.reply_text(
        "🧹 Context cleared!\n\nFresh start 😎"
    )


# =========================
# STATS
# =========================

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_owner(update):

        await update.message.reply_text(
            "❌ Owner-only command."
        )

        return

    text_status = "🟢 OK" if last_success else "🔴 ERROR"

    if photo_tested:
        photo_status = "🟢 OK" if photo_ok else "🔴 ERROR"
    else:
        photo_status = "🟡 Not tested"

    if video_tested:
        video_status = "🟢 OK" if video_ok else "🔴 ERROR"
    else:
        video_status = "🟡 Not tested"

    supabase_status = "🟢 OK"

    error_text = last_error if last_error else "None"

    report = f"""
🩺 H15ai Health Report

🤖 Text AI: {text_status}
🖼️ Photo AI: {photo_status}
🎬 Video AI: {video_status}
🗄️ Supabase: {supabase_status}

📊 Stats
• Messages: {stats["total_messages"]}
• Replies: {stats["total_replies"]}
• Users: {stats["total_users"]}
• Photos: {stats["total_photos"]}
• Videos: {stats["total_videos"]}

⚠️ Runtime errors: {runtime_errors}

🕒 Last success:
{last_success or "None"}

❗ Last error:
{error_text}
"""

    await send_long_message(update, report)


# =========================
# TEXT HANDLER
# =========================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global last_success

    try:

        register_user(update.effective_user.id)

        stats["total_messages"] += 1
        save_stats()

        user_text = update.message.text

        reply = await text_ai(user_text)

        await send_long_message(update, reply)

        stats["total_replies"] += 1
        save_stats()

        record_success()

    except Exception as e:

        record_error(e)

        await update.message.reply_text(
            "⚠️ AI side pe error aa gaya. Thodi der baad try kar."
        )


# =========================
# PHOTO HANDLER
# =========================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global photo_tested, photo_ok

    photo_tested = True

    try:

        register_user(update.effective_user.id)

        stats["total_messages"] += 1
        stats["total_photos"] += 1
        save_stats()

        photo = update.message.photo[-1]

        file = await photo.get_file()

        image_bytes = await file.download_as_bytearray()

        prompt = (
            update.message.caption
            or
            "Analyze this image carefully. Explain what is visible and answer the user's likely question."
        )

        reply = await vision_ai(
            bytes(image_bytes),
            prompt
        )

        await send_long_message(update, reply)

        stats["total_replies"] += 1
        save_stats()

        photo_ok = True
        record_success()

    except Exception as e:

        photo_ok = False
        record_error(e)

        await update.message.reply_text(
            "⚠️ Photo AI side pe error aa gaya."
        )


# =========================
# VIDEO HANDLER
# =========================

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global video_tested, video_ok

    video_tested = True

    try:

        register_user(update.effective_user.id)

        stats["total_messages"] += 1
        stats["total_videos"] += 1
        save_stats()

        video = update.message.video

        file = await video.get_file()

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = os.path.join(
                temp_dir,
                "video.mp4"
            )

            await file.download_to_drive(video_path)

            output_pattern = os.path.join(
                temp_dir,
                "frame_%02d.jpg"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                "fps=1/2,scale=512:-1",
                "-frames:v",
                "4",
                "-q:v",
                "4",
                output_pattern,
            ]

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120
                )

            except asyncio.TimeoutError:

                process.kill()
                await process.wait()

                raise Exception("Video processing timed out")

            frames = []

            for filename in sorted(os.listdir(temp_dir)):

                if filename.endswith(".jpg"):

                    frame_path = os.path.join(
                        temp_dir,
                        filename
                    )

                    with open(frame_path, "rb") as f:
                        frames.append(f.read())

            if not frames:
                raise Exception("No video frames extracted")

            combined = []

            for frame in frames:

                encoded = base64.b64encode(
                    frame
                ).decode("utf-8")

                combined.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}"
                    }
                })

            prompt = (
                update.message.caption
                or
                "Analyze this video using the provided frames. Explain what is happening in the video in chronological order. Mention uncertainty if the sampled frames are insufficient."
            )

            combined.insert(
                0,
                {
                    "type": "text",
                    "text": prompt
                }
            )

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": combined,
                    }
                ],
                max_tokens=950,
                reasoning_effort="none",
            )

            reply = response.choices[0].message.content

            await send_long_message(update, reply)

        stats["total_replies"] += 1
        save_stats()

        video_ok = True
        record_success()

    except Exception as e:

        video_ok = False
        record_error(e)

        await update.message.reply_text(
            "⚠️ Video AI side pe error aa gaya."
        )


# =========================
# ERROR HANDLER
# =========================

async def error_handler(update, context):

    error = context.error

    record_error(error)

    print(
        "Telegram error:",
        repr(error)
    )


# =========================
# FASTAPI
# =========================

@app.get("/")
async def home():

    return {
        "status": "H15ai is running",
        "bot": "H15ai_bot"
    }


@app.post("/webhook")
async def webhook(request: Request):

    data = await request.json()

    update = Update.de_json(
        data,
        telegram_app.bot
    )

    await telegram_app.process_update(update)

    return {
        "ok": True
    }


# =========================
# MAIN
# =========================

async def main():

    global telegram_app

    load_stats()

    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler("start", start)
    )

    telegram_app.add_handler(
        CommandHandler("help", help_command)
    )

    telegram_app.add_handler(
        CommandHandler("about", about)
    )

    telegram_app.add_handler(
        CommandHandler("clear", clear_context)
    )

    telegram_app.add_handler(
        CommandHandler("stats", stats_command)
    )

    telegram_app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    telegram_app.add_handler(
        MessageHandler(
            filters.VIDEO,
            handle_video
        )
    )

    telegram_app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_text
        )
    )

    telegram_app.add_error_handler(
        error_handler
    )

    await telegram_app.initialize()
    await telegram_app.start()

    webhook_url = (
        WEBHOOK_URL.rstrip("/")
        + "/webhook"
    )

    await telegram_app.bot.set_webhook(
        webhook_url
    )

    print(
        "H15ai running:",
        webhook_url
    )

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
