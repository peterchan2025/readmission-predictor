# Readmission Predictor (demo)

This repo contains a demo pipeline that:
- Simulates EHR-like data
- Trains a RandomForest to predict 30-day readmission
- Serves predictions via a Flask API

## Files
- data_simulation_and_train.py : simulate data, train, save model
- serve_model.py : Flask app to serve predictions
- requirements.txt : python dependencies

## Run in VS Code (recommended)
1. Create and activate a virtual environment:
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python data_simulation_and_train.py

   This will create `rf_model.joblib` and `roc_curve.png`.

4. Run the server:
   python serve_model.py

5. Test prediction:
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{\"age\":65,\"sex\":\"F\",\"insurance\":\"Medicare\",\"length_of_stay\":5,\"num_prior_admissions_6mo\":1,\"num_ed_visits_30d\":0,\"num_medications\":8,\"hr_diagnosis_flag\":1,\"last_cr\":1.1,\"social_risk_flag\":0,\"discharge_dest\":\"Home\"}'

## Notes
- This is a demo with synthetic data intended for learning and reporting only.
- For production use: secure the Flask app, use HTTPS, authentication, logging, monitoring, and deploy on a proper server or container.
