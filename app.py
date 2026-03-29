from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model + encoder
model, mlb = joblib.load("model.pkl")

# Load dataset for symptom list
data = pd.read_csv("dataset.csv")
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
    name = request.form.get('name')
    selected_symptoms = [request.form.get(f'symptom{i}') for i in range(1,6)]
    selected_symptoms = [s for s in selected_symptoms if s]

    # Convert symptoms to model input
    X_input = mlb.transform([selected_symptoms])

    # Get probabilities
    probs = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Top 3 diseases
    top_indices = probs.argsort()[-3:][::-1]
    results = []
    for i in top_indices:
        results.append({
            "disease": classes[i],
            "prob": round(probs[i]*100, 2)
        })

    return render_template("index.html",
                           symptoms=symptoms,
                           results=results,
                           name=name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)