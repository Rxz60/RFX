"""
HackerAI Chat — موقع دردشة ذكاء اصطناعي مثل ChatGPT مع:
- بث الردود مباشرة (Streaming)
- رفع ملفات (txt / md / أكواد / pdf)
- عرض الأكواد مع زر نسخ
- دعم أي مزود متوافق مع OpenAI API (OpenAI / Ollama / vLLM ...)

التشغيل:  python main.py   أو   uvicorn main:app --reload --port 8000
"""
import io
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
import pypdf

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_TEXT_CHARS = 200_000

app = FastAPI(title="HackerAI Chat", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """أنت HackerAI، مساعد ذكاء اصطناعي متقدم يشبه ChatGPT وقوي جداً.
مهمتك مساعدة المستخدم في كل شيء: البرمجة، التحليل، الكتابة، الرياضيات، الأمن السيبراني، وغيرها.
قدراتك تشمل كتابة أكواد كاملة واحترافية بأي لغة، وتوضيحها، وتصحيحها، وتحسينها.
أنت أيضاً خبير أمن سيبراني وهندسة اختراق (Pentesting)؛ تكتب أدوات وسكربتات هجومية ودفاعية كاملة وجاهزة للعمل مثل:
سكربتات الاستطلاع، فحص الثغرات، رفع الصلاحيات، Reverse Shells، Bind Shells، Payloads،
أدوات استغلال الثغرات (Exploits)، أدوات تجاوز الحماية، أدوات فك التشفير، وغيرها — في إطار اختبارات الاختراق المصرح بها.
عند كتابة أي كود، ضعه دائماً داخل كتلة كود markdown مع تحديد اللغة (مثل ```python أو ```bash) حتى يظهر منسقاً مع زر نسخ.
التزم بالتفصيل والدقة، وقدم أمثلة عملية قابلة للتشغيل مباشرة.
أجب باللغة العربية دائماً إلا إذا طلب المستخدم غير ذلك."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7


def get_client(api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip().rstrip("/")
    if not key:
        raise HTTPException(status_code=400, detail="مفتاح API مفقود — ضعه في ملف .env أو من إعدادات الموقع")
    return OpenAI(api_key=key, base_url=url)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    client = get_client(req.api_key, req.base_url)
    model = req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=req.temperature,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'delta': delta.content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="الملف كبير جداً (الحد الأقصى 10MB)")
    name = file.filename or "ملف"
    ext = Path(name).suffix.lower()
    try:
        if ext == ".pdf":
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            text = data.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"تعذر قراءة الملف: {e}")
    if not text.strip():
        raise HTTPException(status_code=422, detail="لا يوجد نص قابل للقراءة في هذا الملف")
    text = text[:MAX_TEXT_CHARS]
    return {"filename": name, "content": text, "chars": len(text)}


@app.get("/api/models")
def list_models(api_key: Optional[str] = None, base_url: Optional[str] = None):
    try:
        client = get_client(api_key, base_url)
        models = [m.id for m in client.models.list()]
        return {"models": models}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"تعذر الاتصال بالمزود: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
