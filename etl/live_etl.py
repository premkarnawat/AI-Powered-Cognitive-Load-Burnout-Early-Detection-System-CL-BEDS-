import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os
from datetime import date

load_dotenv()

supabase = create_client(
 os.getenv("SUPABASE_URL"),
 os.getenv("SUPABASE_KEY")
)

data = supabase.table("user_assessments").select("*").execute()
df = pd.DataFrame(data.data)

df.fillna(df.mean(), inplace=True)
df["label"] = df["prediction"].apply(lambda x:1 if x>0.7 else 0)
df["processed_date"] = date.today()

for _,row in df.iterrows():
    supabase.table("warehouse_assessments").insert({
        "sleep":row["sleep"],
        "work_hours":row["work_hours"],
        "study_hours":row["study_hours"],
        "screen_time":row["screen_time"],
        "stress":row["stress"],
        "fatigue":row["fatigue"],
        "label":row["label"],
        "processed_date":row["processed_date"]
    }).execute()

print("Live ETL Completed")
