import os
import asyncio
import base64
import tempfile
import subprocess
import threading

import imageio_ffmpeg
from openai import OpenAI

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from fastapi import FastAPI, Request
import uvicorn


# =========================
# CONFIG
# =========================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"

OWNER_USERNAME = "Harshupadhyay_15"

MAX_HISTORY = 20
MAX_VIDEO_SIZE = 20 * 1024 * 1024
MAX_VIDEO_FRAMES = 6


# =========================
# CLIENTS
# =========================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=120.0,
)

app = FastAPI()


# =========================
# MEMORY
# =========================

user_histories = {}

stats = {
    "total_messages": 0,
    "total_replies": 0,
    "total_users": 0,
    "total_photos": 0,
    "total_videos": 0,
}

known_users = set()

stats_lock = threading.Lock()
stats_row_id = None


# =========================
# SUPABASE HELPERS
# =========================

def supabase_headers():
    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


async def supabase_request(method, url, data=None, headers=None):
    import urllib.request
    import json

    def request():
        req = urllib.request.Request(
            url,
            method=method,
            headers=headers or {},
        )

        if data is not None:
            req.data = json.dumps(data).encode()

        with urllib.request.urlopen(req, timeout=15) as response:
            body = response.read().decode()
            return response.status, body

    return await asyncio.to_thread(request)


async def ensure_stats_row():
    """
    Makes sure exactly one usable bot_stats row is used.
    """

    global stats_row_id

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    if stats_row_id is not None:
        return

    url = f"{SUPABASE_URL}/rest/v1/bot_stats?select=id&order=id.asc&limit=1"

    try:
        status, body = await supabase_request(
            "GET",
            url,
            headers=supabase_headers(),
        )

        if status == 200:
            rows = __import__("json").loads(body)

            if rows:
                stats_row_id = rows[0]["id"]
                return

        # No row → create one
        insert_url = f"{SUPABASE_URL}/rest/v1/bot_stats"

        payload = {
            "total_messages": 0,
            "total_replies": 0,
            "total_users": 0,
            "total_photos": 0,
            "total_videos": 0,
        }

        status, body = await supabase_request(
            "POST",
            insert_url,
            data=payload,
            headers={
                **supabase_headers(),
                "Prefer": "return=representation",
            },
        )

        if status in (200, 201):
            rows = __import__("json").loads(body)

            if rows:
                stats_row_id = rows[0]["id"]

    except Exception as e:
        print("Supabase init error:", e)


async def load_stats():
    global stats

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    await ensure_stats_row()

    if stats_row_id is None:
        return

    url = (
        f"{SUPABASE_URL}/rest/v1/bot_stats"
        f"?id=eq.{stats_row_id}"
    )

    try:
        status, body = await supabase_request(
            "GET",
            url,
            headers=supabase_headers(),
        )

        if status == 200:
            rows = __import__("json").loads(body)

            if rows:
                row = rows[0]

                for key in stats:
                    stats[key] = int(row.get(key, 0) or 0)

    except Exception as e:
        print("Supabase load error:", e)


async def save_stats():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    await ensure_stats_row()

    if stats_row_id is None:
        return

    url = (
        f"{SUPABASE_URL}/rest/v1/bot_stats"
        f"?id=eq.{stats_row_id}"
    )

    payload = dict(stats)

    try:
        status, body = await supabase_request(
            "PATCH",
            url,
            data=payload,
            headers={
                **supabase_headers(),
                "Prefer": "return=minimal",
            },
        )

        if status not in (200, 204):
            print("Supabase save failed:", status, body)

    except Exception as e:
        print("Supabase save error:", e)


async def record_stat(key, amount=1):
    """
    Serializes stats updates so simultaneous messages
    don't overwrite each other's counters.
    """

    with stats_lock:
        stats[key] = stats.get(key, 0) + amount
        snapshot = dict(stats)

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    await ensure_stats_row()

    if stats_row_id is None:
        return

    url = (
        f"{SUPABASE_URL}/rest/v1/bot_stats"
        f"?id=eq.{stats_row_id}"
    )

    try:
        status, body = await supabase_request(
            "PATCH",
            url,
            data=snapshot,
            headers={
                **supabase_headers(),
                "Prefer": "return=minimal",
            },
        )

        if status not in (200, 204):
            print("Stats update failed:", status, body)

    except Exception as e:
        print("Stats update error:", e)


async def register_user(user_id):
    global known_users

    if user_id in known_users:
        return

    known_users.add(user_id)

    await record_stat("total_users")


# =========================
# PERSONALITY
# =========================

SYSTEM_PROMPT = """
You are H15ai, an AI assistant created by Harsh Upadhyay.

PERSONALITY:
- Talk naturally like a smart, chill friend.
- Match the user's language: Hinglish when they use Hinglish,
  English when they use English.
- Do NOT automatically call everyone "bhai", "bro", "bby",
  "didi", etc.
- Mirror the user's style without becoming annoying.
- Be friendly, funny when appropriate, and helpful.
- Never sound unnecessarily formal.
- Avoid filler and unnecessary explanations.

ANSWER STYLE:
- Give the answer directly.
- Keep normal answers concise.
- If the question needs explanation, explain clearly but don't ramble.
- Don't repeat the user's question unnecessarily.
- Don't add unrelated information.
- Don't constantly mention that you are an AI.
- Don't advertise yourself.

ACCURACY:
- Never invent facts.
- If you are unsure, clearly say that you are unsure.
- Do not present guesses as facts.
- For sports/player/current-event information, do not pretend
  you have live internet access.
- If exact/current information cannot be verified, say so.
- For maths, physics, chemistry and calculations:
  solve carefully and re-check the final answer.
- If there are multiple possible interpretations, ask briefly
  or state the assumption.

CREATOR:
- H15ai was created by Harsh Upadhyay as an AI learning project.
- Public creator username: @HARSHUPADHYAY_15
- Do not reveal private information about the creator.

IMAGES:
- When given an image, analyze only what is actually visible.
- Don't invent details.
- For charts/tables, read values carefully.

VIDEOS:
- Video analysis may be based on sampled frames.
- Mention uncertainty if something cannot be determined from
  the available frames.

PRIVACY:
- Don't claim to remember private information unless it is actually
  present in the current conversation context.
"""


# =========================
# AI
# =========================

def get_history(user_id):
    return user_histories.setdefault(user_id, [])


def trim_history(history):
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]


async def ask_ai(user_id, text):
    history = get_history(user_id)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": text,
    })

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1200,
        )

        answer = response.choices[0].message.content

        history.append({
            "role": "user",
            "content": text,
        })

        history.append({
            "role": "assistant",
            "content": answer,
        })

        trim_history(history)

        return answer

    except Exception as e:
        print("AI error:", repr(e))

        return (
            "Abhi AI response mein problem aa gayi 😭 "
            "thodi der baad try kar."
        )


# =========================
# COMMANDS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await register_user(update.effective_user.id)

    await update.message.reply_text(
        "Yo 👋 Main H15ai hoon.\n"
        "Questions, study, coding, ideas, images/video analysis "
        "— jo chahiye pooch."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/start — Start H15ai\n"
        "/help — Help\n"
        "/about — About H15ai\n"
        "/clear — Clear chat memory\n"
        "/ping — Check bot\n"
        "/stats — Bot stats (creator only)"
    )


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 H15ai\n\n"
        "Created by Harsh Upadhyay\n"
        "AI learning project\n"
        "Creator: @HARSHUPADHYAY_15\n\n"
        "Can do:\n"
        "• AI chat & Q&A\n"
        "• Study help\n"
        "• Maths/Physics/Chemistry\n"
        "• Coding\n"
        "• Writing & ideas\n"
        "• Translation & summaries\n"
        "• Image understanding\n"
        "• Basic video understanding\n"
        "• Games & quizzes"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    user_histories[user_id] = []

    await update.message.reply_text(
        "Memory cleared 🧹"
    )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pong 🏓 H15ai is alive.")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.effective_user.username or ""

    if username.lower() != OWNER_USERNAME.lower():
        await update.message.reply_text(
            "Ye command creator-only hai."
        )
        return

    await load_stats()

    await update.message.reply_text(
        "📊 H15ai Stats\n\n"
        f"Messages: {stats['total_messages']}\n"
        f"Replies: {stats['total_replies']}\n"
        f"Users: {stats['total_users']}\n"
        f"Photos: {stats['total_photos']}\n"
        f"Videos: {stats['total_videos']}"
    )


# =========================
# TEXT
# =========================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text

    await register_user(user_id)
    await record_stat("total_messages")

    answer = await ask_ai(user_id, text)

    await update.message.reply_text(answer)

    await record_stat("total_replies")


# =========================
# IMAGE
# =========================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    await register_user(user_id)
    await record_stat("total_messages")
    await record_stat("total_photos")

    try:
        photo = update.message.photo[-1]

        tg_file = await context.bot.get_file(photo.file_id)

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:
            path = temp.name

        await tg_file.download_to_drive(path)

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        os.remove(path)

        prompt = update.message.caption or (
            "Analyze this image carefully and answer what the user "
            "is asking. If no question is given, describe the useful "
            "information visible in the image concisely."
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
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{encoded}"
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=1200,
            reasoning_effort="none",
        )

        answer = response.choices[0].message.content

        await update.message.reply_text(answer)

        await record_stat("total_replies")

    except Exception as e:
        print("Photo error:", repr(e))

        await update.message.reply_text(
            "Image analyse karte time error aa gaya 😭"
        )


# =========================
# VIDEO
# =========================

async def extract_frames(video_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    output_pattern = os.path.join(
        tempfile.gettempdir(),
        "h15ai_frame_%03d.jpg"
    )

    command = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vf",
        "fps=1/2,scale=768:-1",
        "-frames:v",
        str(MAX_VIDEO_FRAMES),
        output_pattern,
    ]

    process = await asyncio.to_thread(
        subprocess.run,
        command,
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr)

    frames = []

    for i in range(1, MAX_VIDEO_FRAMES + 1):
        path = output_pattern.replace(
            "%03d",
            f"{i:03d}"
        )

        if os.path.exists(path):
            frames.append(path)

    return frames


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    await register_user(user_id)
    await record_stat("total_messages")
    await record_stat("total_videos")

    video = update.message.video

    if video.file_size and video.file_size > MAX_VIDEO_SIZE:
        await update.message.reply_text(
            "Video 20MB se chhoti bhejo 👍"
        )
        return

    video_path = None
    frame_paths = []

    try:
        tg_file = await context.bot.get_file(video.file_id)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as temp:
            video_path = temp.name

        await tg_file.download_to_drive(video_path)

        frame_paths = await extract_frames(video_path)

        if not frame_paths:
            await update.message.reply_text(
                "Video ke frames read nahi ho paaye 😭"
            )
            return

        content = []

        prompt = update.message.caption or (
            "Analyze these sampled frames from a video. "
            "Explain what is happening and answer the user's "
            "question if one is given. Be careful not to claim "
            "details that cannot be determined from the frames."
        )

        content.append({
            "type": "text",
            "text": prompt,
        })

        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                encoded = base64.b64encode(
                    f.read()
                ).decode()

            content.append({
                "type": "image_url",
                "image_url": {
                    "url":
                    f"data:image/jpeg;base64,{encoded}"
                },
            })

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
                    "content": content,
                },
            ],
            temperature=0.2,
            max_tokens=1200,
            reasoning_effort="none",
        )

        answer = response.choices[0].message.content

        await update.message.reply_text(answer)

        await record_stat("total_replies")

    except Exception as e:
        print("Video error:", repr(e))

        await update.message.reply_text(
            "Video analyse karte time error aa gaya 😭"
        )

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

        for path in frame_paths:
            if os.path.exists(path):
                os.remove(path)


# =========================
# TELEGRAM APPLICATION
# =========================

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
    CommandHandler("clear", clear_command)
)

telegram_app.add_handler(
    CommandHandler("ping", ping)
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


# =========================
# FASTAPI WEBHOOK
# =========================

@ app.on_event("startup")
async def startup():
    await telegram_app.initialize()
    await telegram_app.start()

    await load_stats()

    webhook = WEBHOOK_URL.rstrip("/") + "/webhook"

    await telegram_app.bot.set_webhook(webhook)

    print("H15ai started.")
    print("Webhook:", webhook)


@ app.on_event("shutdown")
async def shutdown():
    await save_stats()

    await telegram_app.stop()
    await telegram_app.shutdown()


@ app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    update = Update.de_json(
        data,
        telegram_app.bot
    )

    await telegram_app.update_queue.put(update)

    return {"ok": True}


@ app.get("/")
async def root():
    return {
        "status": "online",
        "bot": "H15ai"
    }


# =========================
# RUN
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
