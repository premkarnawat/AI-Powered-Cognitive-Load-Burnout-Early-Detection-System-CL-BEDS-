from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
from explainability.explain import explain_prediction

# =====================================================
# ENV
# =====================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials missing")

if not OPENROUTER_API_KEY:
    raise Exception("OpenRouter API key missing")

# =====================================================
# CLIENTS
# =====================================================

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

llm_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# =====================================================
# MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "burnout_model.pkl")
model = joblib.load(model_path)

# =====================================================
# APP
# =====================================================

app = FastAPI(title="AI Powered Burnout Detection System")

# =====================================================
# SCHEMAS
# =====================================================

class PredictRequest(BaseModel):
    user_id: str
    fatigue: int
    work_hours: int
    sleep: int
    screen_time: int
    study_hours: int
    social_media_hours: int
    stress: int


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatWithReportRequest(BaseModel):
    user_id: str
    message: str


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
def predict(data: PredictRequest):

    features = [
        data.fatigue,
        data.work_hours,
        data.sleep,
        data.screen_time,
        data.study_hours,
        data.social_media_hours,
        data.stress
    ]

    probability = model.predict_proba([features])[0][1]

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Moderate"
    else:
        risk = "Low"

    explanation = explain_prediction(features, model)

    supabase.table("user_assessments").insert({
        "user_id": data.user_id,
        "sleep": data.sleep,
        "work_hours": data.work_hours,
        "study_hours": data.study_hours,
        "screen_time": data.screen_time,
        "social_media_hours": data.social_media_hours,
        "stress": data.stress,
        "fatigue": data.fatigue,
        "burnout_probability": float(probability)
    }).execute()

    return {
        "burnout_probability": float(probability),
        "risk_level": risk,
        "explanation": explanation
    }


# =====================================================
# CHAT
# =====================================================

@app.post("/chat")
def chat(data: ChatRequest):

    try:
        response = llm_client.chat.completions.create(
            model="openchat/openchat-7b:free",
            messages=[
                {"role": "system", "content": "You are a supportive mental health assistant."},
                {"role": "user", "content": data.message}
            ]
        )

        reply = response.choices[0].message.content

        supabase.table("chat_history").insert({
            "user_id": data.user_id,
            "user_message": data.message,
            "bot_reply": reply
        }).execute()

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# CHAT WITH REPORT (RAG)
# =====================================================

@app.post("/chat_with_report")
def chat_with_report(data: ChatWithReportRequest):

    try:
        records = supabase.table("user_assessments") \
            .select("*") \
            .eq("user_id", data.user_id) \
            .execute()

        history = records.data

        prompt = f"""
You are an AI mental wellness analyst.

User burnout history:
{history}

User question:
{data.message}

Analyze patterns and give clear advice.
"""

        response = llm_client.chat.completions.create(
            model="openchat/openchat-7b:free",
            messages=[{"role": "user", "content": prompt}]
        )

        return {"reply": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
