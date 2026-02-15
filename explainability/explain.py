import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd

# Load model
model = joblib.load("model/burnout_model.pkl")

# Load training data for LIME background
data = pd.read_csv("datasets/final_clean_dataset.csv")
X = data.drop("label", axis=1)

explainer = LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=X.columns.tolist(),
    class_names=["No Burnout", "Burnout"],
    mode="classification"
)

def explain_prediction(features):

    exp = explainer.explain_instance(
        np.array(features),
        model.predict_proba,
        num_features=5
    )

    return [item[0] for item in exp.as_list()]
