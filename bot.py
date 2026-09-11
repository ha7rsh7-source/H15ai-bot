from pathlib import Path

code = r'''import os
import asyncio
import base64
import re
import subprocess
import tempfile
from collections import defaultdict, deque

import imageio_ffmpeg
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from fastapi import FastAPI, Request
import uvicorn


# ============================================================
# H15ai v4
# Owner Mode + Health Monitor + Supabase Stats
# No self-writing /mnt/data code
# ============================================================

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
- If someone directly asks who created you, say:
  "Harsh Upadhyay created me as an AI learning project."
- Do not reveal private information about Harsh.
- Do not invent facts about the creator or this project.

TONE:
- Talk naturally like a smart, chill friend.
- Match the user's language and energy.
- Hindi/Hinglish is fine when the user uses it.
- Never automatically call everyone "bhai".
- Use bhai/bro only when it feels natural.
- Do not overuse emojis or filler.
- Be concise unless the question needs detail.

ACCURACY:
- Never invent facts, dates, statistics, names, quotes or sources.
- If you are uncertain, clearly say so.
- Web search is OFF in this bot. Never pretend that you verified something live.
- For current information, say that live verification is unavailable.

MATH / SCIENCE:
- Understand the question first.
- Identify quantities and units.
- Choose the correct formula or principle.
- Solve carefully.
- Recheck arithmetic, signs, units and the final conclusion.
- Do not change a correct answer just because another answer claims it is wrong.
- For physics, distinguish displacement, distance, velocity, speed, etc. carefully.
- For multi-object mechanics, account for all relevant masses and relative displacements.

IMAGES:
- Use only information actually visible in the image.
- Read visible text when possible.
- If something is unclear, say so instead of guessing.
- For a visible question, solve it and explain the result.

VIDEOS:
- Video understanding uses sampled frames.
- Audio is not analyzed.
- Do not claim to have seen every moment of a long video.
- If the sampled frames are insufficient, say so.

PRIVACY:
- You do not have access to a user's private Telegram chats, contacts or account.
- Do not claim to see information that the user has not sent to you.
"""

user_histories = defaultdict(lambda: deque(maxlen=30))

stats = {
    "total_messages": 0,
    "total_replies": 0,
    "total_users": 0,
    "total_photos": 0,
    "total_videos": 0,
}

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

health_lock = asyncio.Lock()


# ============================================================
# Health helpers
# ============================================================

async def set_health_success(service: str):
    async with health_lock:
        health[f"{service}_ok"] = True


async def set_health_error(service: str, exc: Exception):
    async with health_lock:
        health[f"{service}_ok"] = False
        health["last_error"] = str(exc)[:500]
        health["last_error_type"] = type(exc).__name__
        health["error_count"] += 1


async def set_last_success():
    from datetime import datetime, timezone
    async with health_lock:
        health["last_success"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def owner_only(update: Update) -> bool:
    return bool(update.effective_user and update.effective_user.id == OWNER_ID)


def looks_like_health_question(text: str) -> bool:
    t = text.lower().strip()

    phrases = [
        "health check",
        "health status",
        "bot health",
        "bot status",
        "status bata",
        "status bta",
        "status kya",
        "sab sahi",
        "sab theek",
        "sab thik",
        "koi error",
        "error hai",
        "errors hai",
        "error bata",
        "error bta",
        "kya chal raha",
        "kaisa chal raha",
        "kaise chal raha",
        "bot kaisa",
        "bot ka status",
        "system status",
        "system check",
    ]

    return any(p in t for p in phrases)


async def health_report() -> str:
    async with health_lock:
        h = dict(health)
        s = dict(stats)

    def mark(value):
        if value is True:
            return "🟢 OK"
        if value is False:
            return "🔴 ERROR"
        return "🟡 Not tested"

    return (
        "🩺 H15ai Health Report\n\n"
        f"🤖 Text AI: {mark(h['text_ok'])}\n"
        f"🖼️ Photo AI: {mark(h['photo_ok'])}\n"
        f"🎬 Video AI: {mark(h['video_ok'])}\n"
        f"🗄️ Supabase: {mark(h['supabase_ok'])}\n\n"
        "📊 Stats\n"
        f"• Messages: {s['total_messages']}\n"
        f"• Replies: {s['total_replies']}\n"
        f"• Users: {s['total_users']}\n"
        f"• Photos: {s['total_photos']}\n"
        f"• Videos: {s['total_videos']}\n\n"
        f"⚠️ Runtime errors: {h['error_count']}\n"
        f"🕒 Last success: {h['last_success'] or 'None'}\n"
        f"❗ Last error: {h['last_error'] or 'None'}"
    )


# ============================================================
# Supabase REST helpers
# ============================================================

def supabase_enabled():
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)


def supabase_request(method: str, table: str, payload=None):
    import urllib.request

    if not supabase_enabled():
        return None

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    data = None

    if payload is not None:
        import json
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "apikey": SUPABASE_SECRET_KEY,
            "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )

    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read()


async def load_stats():
    if not supabase_enabled():
        await set_health_error("supabase", RuntimeError("SUPABASE_URL or SUPABASE_SECRET_KEY is missing"))
        return

    try:
        import json

        def work():
            raw = supabase_request(
                "GET",
                "bot_stats",
                None,
            )
            if not raw:
                return None
            rows = json.loads(raw.decode("utf-8"))
            return rows[0] if rows else None

        row = await asyncio.to_thread(work)

        if row:
            for key in stats:
                stats[key] = int(row.get(key, 0) or 0)

        await set_health_success("supabase")

    except Exception as exc:
        await set_health_error("supabase", exc)


async def save_stats():
    if not supabase_enabled():
        return

    try:
        payload = dict(stats)

        def work():
            import json
            data = json.dumps(payload).encode("utf-8")

            import urllib.request
            url = f"{SUPABASE_URL}/rest/v1/bot_stats?id=eq.1"

            req = urllib.request.Request(
                url,
                data=data,
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

        await asyncio.to_thread(work)
        await set_health_success("supabase")

    except Exception as exc:
        await set_health_error("supabase", exc)


async def register_user(user_id: int):
    if user_id not in known_users:
        known_users.add(user_id)
        stats["total_users"] += 1


async def record_event(event: str, user_id: int | None = None):
    if user_id is not None:
        await register_user(user_id)

    if event in stats:
        stats[event] += 1

    await save_stats()


# ============================================================
# AI helpers
# ============================================================

def clean_reply(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def call_text_ai(messages):
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return clean_reply(response.choices[0].message.content)


def call_vision_ai(messages):
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    return clean_reply(response.choices[0].message.content)


def image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


# ============================================================
# Telegram commands
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Hey! I'm H15ai.\n\n"
        "A smart AI bot for chatting, studying, coding, ideas, images and more.\n\n"
        "Try asking me anything 😎\n"
        "Use /help to see commands."
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🧠 H15ai Commands\n\n"
        "/start — Start the bot\n"
        "/help — Show help\n"
        "/clear — Clear your chat context\n"
        "/about — About H15ai\n"
        "/stats — Owner stats (owner only)\n\n"
        "You can also send:\n"
        "• Normal messages\n"
        "• Photos/questions\n"
        "• Videos for basic frame-based understanding\n"
    )
    await update.message.reply_text(text)


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 H15ai\n\n"
        "Created by Harsh Upadhyay as an AI learning project.\n"
        f"Creator: {OWNER_USERNAME}\n\n"
        "Built with Telegram + Groq + Render + Supabase."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("🧹 Chat context cleared.")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        await update.message.reply_text("⛔ This command is owner-only.")
        return

    await update.message.reply_text(await health_report())


# ============================================================
# Text chat
# ============================================================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()

    # Owner natural-language health check
    if owner_only(update) and looks_like_health_question(text):
        await update.message.reply_text(await health_report())
        return

    await record_event("total_messages", user_id)

    history = user_histories[user_id]

    history.append({"role": "user", "content": text})

    messages = [{"role": "system", "content": PERSONALITY}]
    messages.extend(history)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        answer = await asyncio.to_thread(call_text_ai, messages)

        if not answer:
            answer = "Hmm, mujhe proper response nahi mila. Ek baar phir try kar 😭"

        history.append({"role": "assistant", "content": answer})

        await update.message.reply_text(answer)
        await record_event("total_replies", None)

        await set_health_success("text")
        await set_last_success()

    except Exception as exc:
        await set_health_error("text", exc)
        await update.message.reply_text(
            "⚠️ AI side pe error aa gaya. Thodi der baad try kar."
        )


# ============================================================
# Photo understanding
# ============================================================

async def photo_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    user_id = update.effective_user.id
    await record_event("total_messages", user_id)
    await record_event("total_photos", None)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)

        image_bytes = await tg_file.download_as_bytearray()
        image_url = image_to_data_url(bytes(image_bytes))

        caption = update.message.caption or "Analyze this image and explain what you can see."

        messages = [
            {"role": "system", "content": PERSONALITY},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ]

        answer = await asyncio.to_thread(call_vision_ai, messages)

        if not answer:
            answer = "Image samajhne mein problem aa gayi. Clear image bhej ke try karo."

        await update.message.reply_text(answer)
        await record_event("total_replies", None)

        await set_health_success("photo")
        await set_last_success()

    except Exception as exc:
        await set_health_error("photo", exc)
        await update.message.reply_text(
            "⚠️ Photo analyze karte time error aa gaya."
        )


# ============================================================
# Video understanding
# ============================================================

def extract_video_frames(video_path: str, output_dir: str, max_frames: int = 6):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    output_pattern = os.path.join(output_dir, "frame_%02d.jpg")

    command = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vf",
        "fps=1/2,scale=768:-1",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        output_pattern,
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
            path = os.path.join(output_dir, name)
            with open(path, "rb") as f:
                frames.append(f.read())

    return frames


async def video_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.video:
        return

    user_id = update.effective_user.id
    await record_event("total_messages", user_id)
    await record_event("total_videos", None)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        video = update.message.video
        tg_file = await context.bot.get_file(video.file_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "input.mp4")
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            await tg_file.download_to_drive(video_path)

            frames = await asyncio.to_thread(
                extract_video_frames,
                video_path,
                frames_dir,
                6,
            )

            if not frames:
                raise RuntimeError("No frames could be extracted from the video.")

            content = [
                {
                    "type": "text",
                    "text": (
                        update.message.caption
                        or
                        "Analyze these sampled frames from the video. "
                        "Describe what is happening and answer any visible question. "
                        "Be clear that the video is being understood from sampled frames."
                    ),
                }
            ]

            for frame in frames:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(frame)
                        },
                    }
                )

            messages = [
                {"role": "system", "content": PERSONALITY},
                {"role": "user", "content": content},
            ]

            answer = await asyncio.to_thread(call_vision_ai, messages)

            if not answer:
                answer = "Video samajhne mein proper response nahi mila."

            await update.message.reply_text(answer)
            await record_event("total_replies", None)

            await set_health_success("video")
            await set_last_success()

    except Exception as exc:
        await set_health_error("video", exc)
        await update.message.reply_text(
            "⚠️ Video analyze karte time error aa gaya. "
            "Shorter/clearer video try karo."
        )


# ============================================================
# FastAPI / Render webhook
# ============================================================

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
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "health": health,
        "stats": stats,
    }


@fastapi_app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


# ============================================================
# Main
# ============================================================

async def main():
    global telegram_app

    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("about", about_command))
    telegram_app.add_handler(CommandHandler("clear", clear_command))
    telegram_app.add_handler(CommandHandler("stats", stats_command))

    telegram_app.add_handler(
        MessageHandler(filters.PHOTO, photo_chat)
    )

    telegram_app.add_handler(
        MessageHandler(filters.VIDEO, video_chat)
    )

    telegram_app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )

    await telegram_app.initialize()
    await telegram_app.start()

    await load_stats()

    if WEBHOOK_URL:
        await telegram_app.bot.set_webhook(
            url=f"{WEBHOOK_URL}/webhook",
            drop_pending_updates=True,
        )
        print(f"H15ai webhook set: {WEBHOOK_URL}/webhook")

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
        if WEBHOOK_URL:
            try:
                await telegram_app.bot.delete_webhook()
            except Exception:
                pass

        await telegram_app.stop()
        await telegram_app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
'''

path = Path("/mnt/data/H15ai_bot_v4.py")
path.write_text(code, encoding="utf-8")

print(f"Created: {path}")
print(f"Lines: {len(code.splitlines())}")
