from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd 

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "ML Model API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["age"]
    salary = data["salary"]

    input_data = pd.DataFrame([[age, salary]], columns=["age", "salary"])
    prediction = model.predict(input_data)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 5000)
