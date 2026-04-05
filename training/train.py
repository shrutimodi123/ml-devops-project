import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/data.csv")

X = df[['age', 'salary']]
y = df['purchased']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")

print("Model saved successfully!")

