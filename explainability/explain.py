import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import os

# -----------------------------
# Load Dataset Safely
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "datasets", "final_clean_dataset.csv")

data = pd.read_csv(data_path)

X = data.drop("label", axis=1)

explainer = LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=X.columns.tolist(),
    class_names=["No Burnout", "Burnout"],
    mode="classification"
)

def explain_prediction(features, model):
    exp = explainer.explain_instance(
        np.array(features),
        model.predict_proba,
        num_features=5
    )

    return [item[0] for item in exp.as_list()]
