"""
serve_model.py
Simple Flask app to serve the trained model with both API and web interface.
Run: python serve_model.py
Example:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"age":65,"sex":"F","insurance":"Medicare","length_of_stay":5,"num_prior_admissions_6mo":1,"num_ed_visits_30d":0,"num_medications":8,"hr_diagnosis_flag":1,"last_cr":1.1,"social_risk_flag":0,"discharge_dest":"Home"}'
"""

from flask import Flask, request, jsonify, render_template
import joblib
import traceback
import pandas as pd

app = Flask(__name__, template_folder="templates")

MODEL_PATH = "rf_model.joblib"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = None

@app.route("/")
def home():
    """Display the web form"""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model()
    try:
        data = request.get_json()
        # Support single prediction (dict) or batch (list)
        single = False
        if isinstance(data, dict):
            single = True
            data = [data]
        X = pd.DataFrame(data)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        results = []
        for p, pr in zip(preds.tolist(), probs.tolist()):
            results.append({"readmit_30d_pred": int(p), "probability": float(pr)})
        if single:
            return jsonify(results[0])
        else:
            return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
