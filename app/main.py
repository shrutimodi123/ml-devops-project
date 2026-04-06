from flask import Flask, request, jsonify
import joblib
import pandas as pd
import redis
import json

app = Flask(__name__)

# Load model
model = joblib.load("model/model.pkl")

# Connect Redis
cache = redis.Redis(host='redis', port=6379)

@app.route("/")
def home():
    return "ML Model API Running with Redis"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["age"]
    salary = data["salary"]

    # Cache key
    key = f"{age}_{salary}"

    # Check cache
    cached = cache.get(key)
    if cached:
        return jsonify({"prediction": int(cached), "source": "cache"})

    # Predict
    input_data = pd.DataFrame([[age, salary]], columns=["age", "salary"])
    prediction = int(model.predict(input_data)[0])

    # Store in cache
    cache.set(key, prediction)

    return jsonify({"prediction": prediction, "source": "model"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)