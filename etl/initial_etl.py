import pandas as pd

# ======================
# EXTRACT
# ======================

burnout = pd.read_csv("datasets/work_from_home_burnout_dataset.csv")
habits = pd.read_csv("datasets/student_habits_performance.csv")

print("Burnout Columns:", burnout.columns)
print("Student Columns:", habits.columns)

# ======================
# TRANSFORM
# ======================

# Select needed columns
burnout = burnout[[
    "sleep_hours",
    "work_hours",
    "screen_time_hours",
    "burnout_score",
    "burnout_risk"
]]

habits = habits[[
    "study_hours_per_day",
    "social_media_hours",
    "mental_health_rating"
]]

# Rename columns
burnout.columns = ["sleep","work_hours","screen_time","fatigue","label"]
habits.columns = ["study_hours","social_media_hours","stress"]

# ✅ Convert burnout_risk text → numeric
burnout["label"] = burnout["label"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# ✅ Fill only numeric columns
burnout[["sleep","work_hours","screen_time","fatigue","label"]] = \
    burnout[["sleep","work_hours","screen_time","fatigue","label"]].fillna(
        burnout[["sleep","work_hours","screen_time","fatigue","label"]].mean()
    )

habits = habits.fillna(habits.mean())

# Merge datasets
final_df = pd.concat([
    burnout[["fatigue","work_hours","sleep","screen_time","label"]],
    habits
], axis=1)

final_df.dropna(inplace=True)

# ======================
# LOAD
# ======================

final_df.to_csv("datasets/final_clean_dataset.csv", index=False)
print("ETL Completed Successfully")
