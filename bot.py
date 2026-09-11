from pathlib import Path

code = '''import os
import asyncio
import base64
import re
import subprocess
import tempfile
from collections import defaultdict, deque
from datetime import datetime, timezone

import imageio_ffmpeg
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from fastapi import FastAPI, Request
import uvicorn


# =========================
# H15ai - clean production version
# =========================

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").rstrip("/")
PORT = int(os.environ.get("PORT", "10000"))

OWNER_ID = 1565428409
OWNER_USERNAME = "@Harshupadhyay_15"

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=120.0,
)

PERSONALITY = """
You are H15ai, a smart, funny and genuinely helpful AI chatbot.

CREATOR:
- You were created by Harsh Upadhyay as an AI learning project.
- Creator's public Telegram username is @Harshupadhyay_15.
- If asked who created you, say: "Harsh Upadhyay created me as an AI learning project."
- Do not reveal private information about the creator.
- Never invent facts about the creator or this project.

TONE:
- Talk naturally like a smart, chill friend.
- Match the user's language and energy.
- Hindi/Hinglish is fine when the user uses it.
- Do not automatically call everyone bhai.
- Use bhai/bro only when natural.
- Do not overuse emojis or filler.
- Be concise unless detail is useful.

ACCURACY:
- Never invent facts, dates, statistics, names, quotes or sources.
- If uncertain, say so.
- Live web search is OFF. Never pretend to have verified current information.

MATH AND SCIENCE:
- Understand the question first.
- Identify quantities and units.
- Select the correct formula/principle.
- Solve carefully.
- Recheck arithmetic, signs, units and the final conclusion.
- Distinguish distance/displacement and speed/velocity carefully.
- For multi-object mechanics, account for all relevant masses and relative displacements.

IMAGES:
- Use only visible information.
- Read visible text when possible.
- If unclear, say so instead of guessing.

VIDEOS:
- Videos are understood from sampled frames.
- Audio is not analyzed.
- Never claim to have seen every moment of a long video.

PRIVACY:
- You cannot access users' private Telegram chats, contacts or accounts.
"""


# Per-user temporary conversation context.
histories = defaultdict(lambda: deque(maxlen=30))

# Runtime counters. Persisted to Supabase when configured.
stats = {
    "total_messages": 0,
    "total_replies": 0,
    "total_users": 0,
    "total_photos": 0,
    "total_videos": 0,
}

# Users seen during this running instance.
known_users = set()

health = {
    "text_ok": None,
    "photo_ok": None,
    "video_ok": None,
    "supabase_ok": None,
    "last_error": "",
    "last_error_type": "",
    "error_count": 0,
    "last_success": "",
}


# =========================
# Health
# =========================

def is_owner(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id == OWNER_ID)


def mark_success(service: str):
    health[f"{service}_ok"] = True
    health["last_success"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def mark_error(service: str, exc: Exception):
    health[f"{service}_ok"] = False
    health["last_error"] = str(exc)[:500]
    health["last_error_type"] = type(exc).__name__
    health["error_count"] += 1


def health_question(text: str) -> bool:
    t = text.lower().strip()
    phrases = (
        "health check", "health status", "bot health", "bot status",
        "status bata", "status bta", "status kya", "sab sahi",
        "sab theek", "sab thik", "koi error", "error hai",
        "errors hai", "error bata", "error bta", "kaisa chal raha",
        "kaise chal raha", "bot kaisa", "bot ka status",
        "system status", "system check"
    )
    return any(p in t for p in phrases)


def health_report() -> str:
    def mark(value):
        if value is True:
            return "🟢 OK"
        if value is False:
            return "🔴 ERROR"
        return "🟡 Not tested"

    return (
        "🩺 H15ai Health Report\\n\\n"
        f"🤖 Text AI: {mark(health['text_ok'])}\\n"
        f"🖼️ Photo AI: {mark(health['photo_ok'])}\\n"
        f"🎬 Video AI: {mark(health['video_ok'])}\\n"
        f"🗄️ Supabase: {mark(health['supabase_ok'])}\\n\\n"
        "📊 Stats\\n"
        f"• Messages: {stats['total_messages']}\\n"
        f"• Replies: {stats['total_replies']}\\n"
        f"• Users: {stats['total_users']}\\n"
        f"• Photos: {stats['total_photos']}\\n"
        f"• Videos: {stats['total_videos']}\\n\\n"
        f"⚠️ Runtime errors: {health['error_count']}\\n"
        f"🕒 Last success: {health['last_success'] or 'None'}\\n"
        f"❗ Last error: {health['last_error'] or 'None'}"
    )


# =========================
# Supabase REST
# =========================

def supabase_enabled():
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)


def supabase_get_stats():
    import json
    import urllib.request

    url = f"{SUPABASE_URL}/rest/v1/bot_stats?id=eq.1&select=*"

    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "apikey": SUPABASE_SECRET_KEY,
            "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        },
    )

    with urllib.request.urlopen(req, timeout=15) as response:
        raw = response.read()

    rows = json.loads(raw.decode("utf-8"))
    return rows[0] if rows else None


def supabase_save_stats():
    import json
    import urllib.request

    url = f"{SUPABASE_URL}/rest/v1/bot_stats?id=eq.1"

    payload = json.dumps(stats).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        method="PATCH",
        headers={
            "apikey": SUPABASE_SECRET_KEY,
            "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )

    with urllib.request.urlopen(req, timeout=15) as response:
        return response.status


async def load_stats():
    if not supabase_enabled():
        health["supabase_ok"] = False
        health["last_error"] = "Supabase environment variables are missing."
        health["last_error_type"] = "ConfigurationError"
        return

    try:
        row = await asyncio.to_thread(supabase_get_stats)

        if row:
            for key in stats:
                stats[key] = int(row.get(key, 0) or 0)

        health["supabase_ok"] = True

    except Exception as exc:
        mark_error("supabase", exc)


async def save_stats():
    if not supabase_enabled():
        return

    try:
        await asyncio.to_thread(supabase_save_stats)
        health["supabase_ok"] = True
    except Exception as exc:
        mark_error("supabase", exc)


async def event(name: str, user_id=None):
    if user_id is not None and user_id not in known_users:
        known_users.add(user_id)
        stats["total_users"] += 1

    if name in stats:
        stats[name] += 1

    await save_stats()


# =========================
# AI
# =========================

def clean_reply(text) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def text_ai(messages):
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return clean_reply(response.choices[0].message.content)


def vision_ai(messages):
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    return clean_reply(response.choices[0].message.content)


def data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")


# =========================
# Commands
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hey! I'm H15ai.\\n\\n"
        "AI chat, study help, coding, ideas, image understanding and basic video understanding.\\n\\n"
        "Use /help for commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 H15ai Commands\\n\\n"
        "/start — Start\\n"
        "/help — Help\\n"
        "/clear — Clear your chat context\\n"
        "/about — About H15ai\\n"
        "/stats — Owner health + stats (owner only)\\n\\n"
        "You can also send normal messages, photos and videos."
    )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 H15ai\\n\\n"
        "Created by Harsh Upadhyay as an AI learning project.\\n"
        f"Creator: {OWNER_USERNAME}\\n\\n"
        "Built with Telegram + Groq + Render + Supabase."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    histories.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 Chat context cleared.")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        await update.message.reply_text("⛔ This command is owner-only.")
        return

    await update.message.reply_text(health_report())


# =========================
# Text
# =========================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()

    if is_owner(update) and health_question(text):
        await update.message.reply_text(health_report())
        return

    await event("total_messages", user_id)

    history = histories[user_id]
    history.append({"role": "user", "content": text})

    messages = [{"role": "system", "content": PERSONALITY}]
    messages.extend(history)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        answer = await asyncio.to_thread(text_ai, messages)
        answer = answer or "Mujhe proper response nahi mila. Ek baar phir try kar 😭"

        history.append({"role": "assistant", "content": answer})

        await update.message.reply_text(answer)
        await event("total_replies")
        mark_success("text")

    except Exception as exc:
        mark_error("text", exc)
        await update.message.reply_text(
            "⚠️ AI side pe error aa gaya. Thodi der baad try kar."
        )


# =========================
# Photo
# =========================

async def photo_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    user_id = update.effective_user.id
    await event("total_messages", user_id)
    await event("total_photos")

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await tg_file.download_as_bytearray())

        caption = update.message.caption or "Analyze this image and explain what you can see."

        messages = [
            {"role": "system", "content": PERSONALITY},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption},
                    {"type": "image_url", "image_url": {"url": data_url(image_bytes)}},
                ],
            },
        ]

        answer = await asyncio.to_thread(vision_ai, messages)
        answer = answer or "Image samajhne mein problem aa gayi."

        await update.message.reply_text(answer)
        await event("total_replies")
        mark_success("photo")

    except Exception as exc:
        mark_error("photo", exc)
        await update.message.reply_text("⚠️ Photo analyze karte time error aa gaya.")


# =========================
# Video
# =========================

def extract_frames(video_path: str, output_dir: str, max_frames=6):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = os.path.join(output_dir, "frame_%02d.jpg")

    command = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-vf", "fps=1/2,scale=768:-1",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        pattern,
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=120,
    )

    frames = []
    for name in sorted(os.listdir(output_dir)):
        if name.endswith(".jpg"):
            with open(os.path.join(output_dir, name), "rb") as f:
                frames.append(f.read())

    return frames


async def video_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.video:
        return

    user_id = update.effective_user.id
    await event("total_messages", user_id)
    await event("total_videos")

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        tg_file = await context.bot.get_file(update.message.video.file_id)

        with tempfile.TemporaryDirectory() as temp:
            video_path = os.path.join(temp, "input.mp4")
            frames_dir = os.path.join(temp, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            await tg_file.download_to_drive(video_path)

            frames = await asyncio.to_thread(
                extract_frames, video_path, frames_dir, 6
            )

            if not frames:
                raise RuntimeError("No video frames could be extracted.")

            content = [{
                "type": "text",
                "text": (
                    update.message.caption
                    or
                    "Analyze these sampled frames from the video. "
                    "Explain what is happening. Do not claim to see "
                    "parts that are not represented by the sampled frames."
                ),
            }]

            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url(frame)},
                })

            messages = [
                {"role": "system", "content": PERSONALITY},
                {"role": "user", "content": content},
            ]

            answer = await asyncio.to_thread(vision_ai, messages)
            answer = answer or "Video samajhne mein proper response nahi mila."

            await update.message.reply_text(answer)
            await event("total_replies")
            mark_success("video")

    except Exception as exc:
        mark_error("video", exc)
        await update.message.reply_text(
            "⚠️ Video analyze karte time error aa gaya. Shorter/clearer video try karo."
        )


# =========================
# FastAPI / Render
# =========================

fastapi_app = FastAPI()
telegram_app = None


@fastapi_app.get("/")
async def root():
    return {"status": "H15ai is running"}


@fastapi_app.get("/health")
async def web_health():
    return {
        "status": "ok",
        "bot": "H15ai",
        "health": health,
        "stats": stats,
    }


@fastapi_app.post("/webhook")
async def webhook(request: Request):
    if telegram_app is None:
        return {"ok": False, "error": "Telegram application not initialized"}

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


# =========================
# Main
# =========================

async def main():
    global telegram_app

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("about", about_command))
    telegram_app.add_handler(CommandHandler("clear", clear_command))
    telegram_app.add_handler(CommandHandler("stats", stats_command))

    telegram_app.add_handler(MessageHandler(filters.PHOTO, photo_chat))
    telegram_app.add_handler(MessageHandler(filters.VIDEO, video_chat))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    await telegram_app.initialize()
    await telegram_app.start()

    await load_stats()

    if WEBHOOK_URL:
        await telegram_app.bot.set_webhook(
            url=f"{WEBHOOK_URL}/webhook",
            drop_pending_updates=True,
        )
        print(f"Webhook set: {WEBHOOK_URL}/webhook")

    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        try:
            await telegram_app.bot.delete_webhook()
        except Exception:
            pass

        await telegram_app.stop()
        await telegram_app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
'''

path = Path("/mnt/data/H15ai_bot_CLEAN.py")
path.write_text(code, encoding="utf-8")

# Syntax check before giving it to the user.
compile(code, "bot.py", "exec")

print(f"READY: {path}")
print(f"Lines: {len(code.splitlines())}")
print("Python syntax check: PASSED")
print("No /mnt/data self-writing code is present.")

