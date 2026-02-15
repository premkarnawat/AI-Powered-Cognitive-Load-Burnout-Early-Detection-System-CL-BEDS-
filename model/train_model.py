import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("datasets/final_clean_dataset.csv")

X = df.drop("label",axis=1)
y = df["label"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = XGBClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,pred))

joblib.dump(model,"model/burnout_model.pkl")
print("Model Saved")
