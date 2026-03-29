from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model + dataset safely
try:
    model, mlb = joblib.load("model.pkl")
    data = pd.read_csv("dataset.csv")
except Exception as e:
    print("Error loading model or dataset:", e)

# Prepare symptoms list
symptoms = set()
for col in data.columns[1:-1]:
    for val in data[col].dropna():
        symptoms.add(val.strip())
symptoms = sorted(symptoms)

@app.route('/')
def home():
    return render_template("index.html", symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name')
        selected = [request.form.get(f'symptom{i}') for i in range(1,6)]
        selected = [s for s in selected if s]

        X = mlb.transform([selected])
        probs = model.predict_proba(X)[0]
        top = probs.argsort()[-3:][::-1]

        results = [{"disease": model.classes_[i], "prob": round(probs[i]*100,2)} for i in top]
        return render_template("index.html", symptoms=symptoms, results=results, name=name)
    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)