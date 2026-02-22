from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import traceback
import re
import sqlite3
from datetime import datetime
from flask import Response

load_dotenv()

app = Flask(__name__)
CORS(app)
rag_init()
# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(BASE_DIR, "frontend", "index.html")

# ----------------------------
# GROQ CONFIG
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]

# ----------------------------
# ROW PERSONALITY
# ----------------------------
SYSTEM_PROMPT = """
You are Row, a futuristic high-tech AI assistant.

You are extremely intelligent, concise, accurate, and professional.

Your mission:
- Answer like a top-tier AI assistant.
- Follow user instructions strictly.
- Use conversation context and memory.
- No unnecessary explanations unless asked.
- Keep responses clean, modern, and efficient.

SPECIAL OUTPUT RULES (IMPORTANT):
1. If the user says "code only" or "only code" or "give me code only":
   - Output ONLY code.
   - No markdown explanation.
   - No steps.
   - No bullet points.
   - No intro text.
   - Just code.
   - Add ONE respectful final line comment inside the code like:
     # Respectfully, Row

2. If the user asks for code:
   - Provide code first.
   - Keep explanation minimal unless asked.

3. Search mode:
   - bullet points only
   - factual and practical
   - no hallucinated sources

DEFAULT RULE:
- Keep answers compact but powerful.
"""

# ----------------------------
# MEMORY (IN-RAM CHAT HISTORY)
# ----------------------------
MEMORY = {}
MAX_MEMORY_MESSAGES = 20  # last 20 messages total (user+assistant)

# ----------------------------
# RAG (SQLite FTS5) - Lightweight Retrieval
# ----------------------------
RAG_DB_PATH = os.path.join(os.path.dirname(__file__), "rag.db")

def rag_init():
    con = sqlite3.connect(RAG_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS rag_docs
        USING fts5(title, content, source, created_at);
    """)
    con.commit()
    con.close()

def rag_add_doc(title: str, content: str, source: str = "manual"):
    title = (title or "").strip()[:200]
    content = (content or "").strip()
    if not content:
        return False

    con = sqlite3.connect(RAG_DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO rag_docs(title, content, source, created_at) VALUES(?,?,?,?)",
        (title or "Untitled", content, source, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()
    return True

def rag_search(query: str, k: int = 5):
    query = (query or "").strip()
    if not query:
        return []

    con = sqlite3.connect(RAG_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT title, source, substr(content, 1, 900) as snippet
        FROM rag_docs
        WHERE rag_docs MATCH ?
        LIMIT ?;
    """, (query, int(k)))

    rows = cur.fetchall()
    con.close()

    return [{"title": t, "source": s, "snippet": sn} for (t, s, sn) in rows]

def wants_rag(user_message: str) -> bool:
    t = (user_message or "").lower()
    return (
        "search my docs" in t or
        "search my notes" in t or
        "from my docs" in t or
        "in my notes" in t or
        "use my docs" in t
    )
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts).strip()
    except Exception as e:
        return ""

# ----------------------------
# IMAGE REQUEST HANDLING (polite refusal + prompt)
# ----------------------------
IMAGE_KEYWORDS = [
    "generate an image", "create an image", "make an image", "draw",
    "image of", "picture of", "photo of", "render", "illustration",
    "logo", "poster", "thumbnail", "wallpaper"
]

def is_image_request(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in IMAGE_KEYWORDS)

def guess_image_size(user_text: str) -> str:
    t = (user_text or "").lower()
    m = re.search(r"\b(\d{3,4})\s*x\s*(\d{3,4})\b", t)
    if m:
        return f"{m.group(1)}x{m.group(2)}"
    if "16:9" in t or "1920x1080" in t or "1080p" in t or "widescreen" in t:
        return "1792x1024 (16:9)"
    if "9:16" in t or "portrait" in t or "reel" in t or "shorts" in t:
        return "1024x1792 (9:16)"
    if "4:3" in t:
        return "1024x768 (4:3)"
    if "thumbnail" in t:
        return "1280x720 (YouTube thumbnail)"
    if "wallpaper" in t:
        return "1920x1080 (desktop wallpaper)"
    return "1024x1024"

def build_image_prompt(user_text: str) -> str:
    size = guess_image_size(user_text)
    return f"""Here’s a premium prompt you can paste into ChatGPT / Gemini / Midjourney / any image generator:

PROMPT:
Create a high-quality image based on: "{user_text}"

STYLE:
Cinematic lighting, ultra-detailed, clean composition, sharp focus, high contrast, realistic textures (unless user asks otherwise)

COMPOSITION:
Clear main subject, uncluttered background, strong depth, professional framing

SIZE:
{size}

NEGATIVE PROMPT:
blurry, low-resolution, distortion, watermark, text artifacts, ugly, noisy background, extra limbs, deformed hands, bad anatomy
"""

# ----------------------------
# AUTO MODE INFERENCE
# ----------------------------
def infer_mode(user_message: str) -> str:
    t = (user_message or "").lower()

    # explicit mode commands
    if re.search(r"\b(code mode|switch to code|mode to code|change mode to code)\b", t): return "Code"
    if re.search(r"\b(search mode|switch to search|mode to search|change mode to search)\b", t): return "Search"
    if re.search(r"\b(plan mode|switch to plan|mode to plan|change mode to plan)\b", t): return "Plan"
    if re.search(r"\b(create mode|switch to create|mode to create|change mode to create)\b", t): return "Create"
    if re.search(r"\b(chat mode|switch to chat|mode to chat|change mode to chat)\b", t): return "Chat"

    # heuristic
    if re.search(r"\b(plan|roadmap|steps|timeline|strategy)\b", t): return "Plan"
    if re.search(r"\b(write code|python|c\+\+|java|javascript|flask|bug|error|traceback|fix|exception)\b", t): return "Code"
    if re.search(r"\b(search|latest|facts|compare|pros and cons|list)\b", t): return "Search"
    if re.search(r"\b(write a|create|generate|caption|post|story|script|copywriting)\b", t): return "Create"
    return "Chat"

# ----------------------------
# HELPERS
# ----------------------------
def clean_output(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def trim_memory(history):
    if len(history) > MAX_MEMORY_MESSAGES:
        return history[-MAX_MEMORY_MESSAGES:]
    return history

def memory_count_for(session_id: str) -> int:
    h = MEMORY.get(session_id, [])
    return len(h) // 2

def groq_chat(messages):
    if not GROQ_API_KEY:
        raise Exception("❌ GROQ_API_KEY missing. Add it in backend/.env file")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None

    for model in GROQ_MODELS:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 1200
            }

            response = requests.post(GROQ_URL, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                output = data["choices"][0]["message"]["content"].strip()
                return output, model
            else:
                last_error = f"Groq API Error {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

    raise Exception("❌ All Groq models failed. Last error: " + str(last_error))

def generate_response(session_id, user_message, mode):
    if session_id not in MEMORY:
        MEMORY[session_id] = []

    history = MEMORY[session_id]

    mode_prompt = f"""
Current Mode: {mode}

STRICT MODE BEHAVIOR:

Chat:
- normal response
- professional
- concise

Search:
- bullet points
- factual summary
- practical suggestions
- no hallucination

Create:
- creative output

Code:
- clean correct code
- minimal explanation unless asked

Plan:
- structured plan
- timeline if needed

IMPORTANT:
Follow the user's formatting instructions strictly.
Use the conversation history as context.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": mode_prompt},
    ]

    # History
    messages.extend(history)

    # Tool: RAG injection (only when user asks)
    if wants_rag(user_message):
        hits = rag_search(user_message, k=5)
        if hits:
            ctx = "RAG CONTEXT (your docs):\n"
            for i, h in enumerate(hits, 1):
                ctx += f"{i}) {h['title']} [{h['source']}]\n{h['snippet']}\n\n"
            messages.append({"role": "system", "content": ctx})

    # New user message
    messages.append({"role": "user", "content": user_message})

    reply, used_model = groq_chat(messages)
    reply = clean_output(reply)

    # Save memory
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    MEMORY[session_id] = trim_memory(history)

    return reply, used_model

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET"])
def serve_frontend():
    if os.path.exists(INDEX_PATH):
        return send_file(INDEX_PATH)
    return "❌ index.html not found. Put it in RowAI/frontend/index.html", 404

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Row backend is online (Groq + Memory + RAG + Auto-Mode)",
        "frontend_found": os.path.exists(INDEX_PATH),
        "models_available": GROQ_MODELS
    })

@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    MEMORY[session_id] = []
    return jsonify({"status": "Memory reset successful", "session_id": session_id})

@app.route("/api/rag/add", methods=["POST"])
def rag_add():
    try:
        data = request.get_json() or {}
        title = data.get("title", "Row Doc")
        content = data.get("content", "")
        source = data.get("source", "manual")
        ok = rag_add_doc(title, content, source)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/rag/search", methods=["POST"])
def rag_search_api():
    try:
        data = request.get_json() or {}
        query = data.get("query", "")
        k = int(data.get("k", 5))
        results = rag_search(query, k)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 500
@app.route("/api/rag/list", methods=["GET"])
def rag_list():
    try:
        q = request.args.get("q", "").strip()
        limit = int(request.args.get("limit", "50"))

        con = sqlite3.connect(RAG_DB_PATH)
        cur = con.cursor()

        if q:
            cur.execute("""
                SELECT rowid, title, source, created_at, substr(content, 1, 240) as snippet
                FROM rag_docs
                WHERE rag_docs MATCH ?
                LIMIT ?;
            """, (q, limit))
        else:
            cur.execute("""
                SELECT rowid, title, source, created_at, substr(content, 1, 240) as snippet
                FROM rag_docs
                ORDER BY rowid DESC
                LIMIT ?;
            """, (limit,))

        rows = cur.fetchall()
        con.close()

        items = []
        for rowid, title, source, created_at, snippet in rows:
            items.append({
                "id": rowid,
                "title": title,
                "source": source,
                "created_at": created_at,
                "snippet": snippet
            })

        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"items": [], "error": str(e)}), 500


@app.route("/api/rag/delete", methods=["POST"])
def rag_delete():
    try:
        data = request.get_json() or {}
        doc_id = int(data.get("id"))

        con = sqlite3.connect(RAG_DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM rag_docs WHERE rowid = ?;", (doc_id,))
        con.commit()
        con.close()

        return jsonify({"ok": True, "deleted": doc_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/rag/clear", methods=["POST"])
def rag_clear():
    try:
        con = sqlite3.connect(RAG_DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM rag_docs;")
        con.commit()
        con.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/rag/export", methods=["GET"])
def rag_export():
    try:
        con = sqlite3.connect(RAG_DB_PATH)
        cur = con.cursor()
        cur.execute("""
            SELECT rowid, title, content, source, created_at
            FROM rag_docs
            ORDER BY rowid DESC;
        """)
        rows = cur.fetchall()
        con.close()

        payload = []
        for rowid, title, content, source, created_at in rows:
            payload.append({
                "id": rowid,
                "title": title,
                "content": content,
                "source": source,
                "created_at": created_at
            })

        return jsonify({"exported_at": datetime.utcnow().isoformat(), "docs": payload})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/upload", methods=["POST"])
def rag_upload():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file uploaded"}), 400

        f = request.files["file"]
        filename = (f.filename or "document").strip()
        raw = f.read()

        # PDF only in this version
        text = extract_text_from_pdf(raw)

        if not text:
            return jsonify({"ok": False, "error": "Could not read PDF text. (Try a text-based PDF)"}), 400

        ok = rag_add_doc(title=filename, content=text, source="pdf_upload")
        return jsonify({"ok": ok, "title": filename, "chars": len(text)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        mode = data.get("mode", "Chat").strip()
        session_id = data.get("session_id", "default")

        if not user_message:
            return jsonify({"reply": "⚠️ Please type something."})

        # Auto mode
        if mode.lower() == "auto":
            mode = infer_mode(user_message)

        # Image requests: polite refusal + prompt (no model call)
        if is_image_request(user_message):
            reply = (
                "⚠️ I can’t generate images directly inside Row yet.\n\n"
                "But I *can* help you generate it on ChatGPT / Gemini / other image tools.\n\n"
                + build_image_prompt(user_message)
            )

            if session_id not in MEMORY:
                MEMORY[session_id] = []
            MEMORY[session_id].append({"role": "user", "content": user_message})
            MEMORY[session_id].append({"role": "assistant", "content": reply})
            MEMORY[session_id] = trim_memory(MEMORY[session_id])

            return jsonify({
                "reply": reply,
                "mode": mode,
                "model_used": "image_prompt_mode",
                "session_id": session_id,
                "memory_count": memory_count_for(session_id)
            })

        reply, used_model = generate_response(session_id, user_message, mode)

        return jsonify({
            "reply": reply,
            "mode": mode,
            "model_used": used_model,
            "session_id": session_id,
            "memory_count": memory_count_for(session_id)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "reply": "⚠️ Backend crashed. Check Flask terminal.",
            "error": str(e)
        }), 500

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    print("✅ Row Backend Starting (Groq + Memory + RAG + Auto-Mode)...")
    print("📌 Frontend Path:", INDEX_PATH)
    print("📌 Frontend Found:", os.path.exists(INDEX_PATH))
    print("📌 Groq Models:", GROQ_MODELS)
    print("📌 RAG DB:", RAG_DB_PATH)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
