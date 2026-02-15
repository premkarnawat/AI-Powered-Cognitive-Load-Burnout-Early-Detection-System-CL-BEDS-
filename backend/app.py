from fastapi import FastAPI
import numpy as np
import joblib
import os
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
from explainability.explain import explain_prediction

# ----------------------------------
# Load Environment Variables
# ----------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ----------------------------------
# Clients
# ----------------------------------

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ----------------------------------
# Load ML Model
# ----------------------------------

model = joblib.load("model/burnout_model.pkl")

# ----------------------------------
# App Init
# ----------------------------------

app = FastAPI(title="AI Powered Burnout Detection System")

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: dict):

    features = [
        data["fatigue"],
        data["work_hours"],
        data["sleep"],
        data["screen_time"],
        data["study_hours"],
        data["social_media_hours"],
        data["stress"]
    ]

    probability = model.predict_proba([features])[0][1]

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Moderate"
    else:
        risk = "Low"

    explanation = explain_prediction(features)

    # Store into Supabase
    supabase.table("user_predictions").insert({
        "user_id": data.get("user_id", "anonymous"),
        "fatigue": data["fatigue"],
        "work_hours": data["work_hours"],
        "sleep": data["sleep"],
        "screen_time": data["screen_time"],
        "study_hours": data["study_hours"],
        "social_media_hours": data["social_media_hours"],
        "stress": data["stress"],
        "burnout_probability": float(probability),
        "risk_level": risk
    }).execute()

    return {
        "burnout_probability": float(probability),
        "risk_level": risk,
        "explanation": explanation
    }

# =========================================================
# SIMPLE CHATBOT
# =========================================================

@app.post("/chat")
def chat(data: dict):

    try:
        response = llm_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a supportive mental health assistant."
                },
                {
                    "role": "user",
                    "content": data["message"]
                }
            ]
        )

        return {"reply": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# CHATBOT WITH USER HISTORY (RAG)
# =========================================================

@app.post("/chat_with_report")
def chat_with_report(data: dict):

    try:
        user_id = data["user_id"]
        question = data["message"]

        records = supabase.table("user_predictions") \
            .select("*") \
            .eq("user_id", user_id) \
            .execute()

        history = records.data

        prompt = f"""
You are an AI mental wellness analyst.

Here is user's burnout history:
{history}

User question:
{question}

Analyze patterns and give clear advice.
"""

        response = llm_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}]
        )

        return {"reply": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}
