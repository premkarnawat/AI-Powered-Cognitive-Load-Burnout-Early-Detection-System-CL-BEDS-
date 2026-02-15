import pandas as pd
from supabase import create_client
from xgboost import XGBClassifier
import joblib, os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY"))

data = supabase.table("warehouse_assessments").select("*").execute()
df = pd.DataFrame(data.data)

X = df.drop(["label","processed_date"],axis=1)
y = df["label"]

model = XGBClassifier()
model.fit(X,y)

joblib.dump(model,"model/burnout_model.pkl")
print("Model Retrained")
