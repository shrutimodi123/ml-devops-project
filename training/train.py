import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv("data/data.csv")

X = df[["age", "salary"]]
y = df["purchased"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")

print("Model trained and saved!")