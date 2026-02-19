from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ProfileRequest(BaseModel):
    user_id: str
    full_name: str | None = None
    phone: str | None = None
    role: str | None = None
    age: int | None = None
    gender: str | None = None
    address: str | None = None
    city: str | None = None
    country: str | None = None

class HealthProfileRequest(BaseModel):
    user_id: str
    height: float | None = None
    weight: float | None = None
    blood_group: str | None = None
    bp: str | None = None
    pulse: int | None = None
    hb: float | None = None
    health_conditions: str | None = None

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================================
# SAVE PROFILE
# =====================================================

@app.post("/save_profile")
def save_profile(data: ProfileRequest):
    try:
        supabase.table("profiles").upsert({
            "id": data.user_id,
            "full_name": data.full_name,
            "phone": data.phone,
            "role": data.role,
            "age": data.age,
            "gender": data.gender,
            "address": data.address,
            "city": data.city,
            "country": data.country
        }).execute()

        return {"message": "Profile saved"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# SAVE HEALTH PROFILE
# =====================================================

@app.post("/save_health_profile")
def save_health_profile(data: HealthProfileRequest):
    try:
        supabase.table("health_profiles").upsert({
            "user_id": data.user_id,
            "height": data.height,
            "weight": data.weight,
            "blood_group": data.blood_group,
            "bp": data.bp,
            "pulse": data.pulse,
            "hb": data.hb,
            "health_conditions": data.health_conditions
        }).execute()

        return {"message": "Health profile saved"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# ✅ FIXED PREDICT
# =====================================================

@app.post("/predict")
def predict(data: PredictRequest):

    # ✅ Force numbers (prevents NaN)
    fatigue = int(data.fatigue)
    work_hours = int(data.work_hours)
    sleep = int(data.sleep)
    screen_time = int(data.screen_time)
    study_hours = int(data.study_hours)
    social_media_hours = int(data.social_media_hours)
    stress = int(data.stress)

    fatigue = int(data.fatigue) * 10   # scale to 0–100
    stress = int(data.stress)          # already correct

features = [
    fatigue,
    data.work_hours,
    data.sleep,
    data.screen_time,
    data.study_hours,
    data.social_media_hours,
    stress
]


    probability = float(model.predict_proba([features])[0][1])

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Moderate"
    else:
        risk = "Low"

    explanation = explain_prediction(features, model)

    supabase.table("user_assessments").insert({
        "user_id": data.user_id,
        "sleep": sleep,
        "work_hours": work_hours,
        "study_hours": study_hours,
        "screen_time": screen_time,
        "social_media_hours": social_media_hours,
        "stress": stress,
        "fatigue": fatigue,
        "burnout_probability": probability,
        "risk_level": risk
    }).execute()

    return {
        "burnout_probability": probability,
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
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "You are a supportive mental health assistant."},
                {"role": "user", "content": data.message}
            ],
            timeout=60
        )

        reply = response.choices[0].message.content or "I'm here to help you."

        supabase.table("chat_history").insert({
            "user_id": data.user_id,
            "user_message": data.message,
            "bot_reply": reply
        }).execute()

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# CHAT WITH REPORT
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
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            timeout=60
        )

        return {"reply": response.choices[0].message.content or "No data yet."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
