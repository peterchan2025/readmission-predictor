
"""

data_simulation_and_train.py
- Simulates EHR-like data
- Trains a RandomForest classifier to predict 30-day readmission
- Saves model to disk as 'rf_model.joblib'
- Outputs test metrics and confusion matrix
Run: python data_simulation_and_train.py
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
def simulate_data(n=5000):
    """

    Simulate a tabular dataset with features typical for readmission prediction.
    """

    df = pd.DataFrame()
    # Demographics
    df['age'] = np.random.randint(18, 95, size=n)
    df['sex'] = np.random.choice(['M', 'F'], size=n, p=[0.48, 0.52])
    df['insurance'] = np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'],
                                       size=n, p=[0.4, 0.2, 0.35, 0.05])
    # Clinical
    df['length_of_stay'] = np.clip(np.random.exponential(scale=3.0, size=n).round(), 1, None)
    df['num_prior_admissions_6mo'] = np.random.poisson(0.4, size=n)
    df['num_ed_visits_30d'] = np.random.poisson(0.2, size=n)
    df['num_medications'] = np.random.randint(1, 20, size=n)
    # High-risk diagnosis flag (heart failure, COPD)
    df['hr_diagnosis_flag'] = np.random.binomial(1, 0.15, size=n)
    # Lab — use normal but inject missing at random
    df['last_cr'] = np.random.normal(1.0, 0.5, size=n)  # creatinine
    mask = np.random.rand(n) < 0.1
    df.loc[mask, 'last_cr'] = np.nan
    # Social risk proxy
    df['social_risk_flag'] = np.random.binomial(1, 0.12, size=n)
    # Discharge destination
    df['discharge_dest'] = np.random.choice(
        ['Home', 'SNF', 'Rehab', 'HomeWithCare'], size=n, p=[0.7, 0.08, 0.05, 0.17]
    )

    # Generate outcome (readmission) with some dependence on features
    logits = (
        0.03 * (df['age'] - 50)
        + 0.5 * df['hr_diagnosis_flag']
        + 0.15 * df['num_prior_admissions_6mo']
        + 0.2 * df['num_ed_visits_30d']
        + 0.05 * (df['length_of_stay'] - 3)
        + 0.35 * df['social_risk_flag']
        + 0.01 * df['num_medications']
    )
    probs = 1 / (1 + np.exp(-logits))
    # Add some randomness
    probs = probs * 0.6 + 0.2 * np.random.rand(n)
    df['readmit_30d'] = (np.random.rand(n) < probs).astype(int)
    return df

def build_pipeline(categorical_features, numeric_features):
    """
    Build a sklearn Pipeline for preprocessing and model training.
    """

    numeric_transformer = SimpleImputer(strategy='median')

    from sklearn.pipeline import Pipeline as SKPipeline
    categorical_transformer = SKPipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
       ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),

    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    pipeline = SKPipeline(steps=[('preproc', preprocessor), ('clf', clf)])
    return pipeline

def main():
    print("Simulating data...")
    df = simulate_data(n=4000)
    features = [
        'age', 'sex', 'insurance', 'length_of_stay', 'num_prior_admissions_6mo',
        'num_ed_visits_30d', 'num_medications', 'hr_diagnosis_flag', 'last_cr',
        'social_risk_flag', 'discharge_dest'
    ]
    target = 'readmit_30d'
    X = df[features]
    y = df[target]

    # Stratified split for demo
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    categorical_features = ['sex', 'insurance', 'discharge_dest']
    numeric_features = [f for f in features if f not in categorical_features]

    pipeline = build_pipeline(categorical_features, numeric_features)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Save the trained pipeline
    model_path = "rf_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    # Evaluate on test set
    print("Evaluating on test set...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Test AUC: {:.3f}".format(auc))
    print("Precision: {:.3f}".format(prec))
    print("Recall: {:.3f}".format(rec))
    print("Confusion Matrix:\n", cm)
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Plot and save ROC curve for the PDF/report
    try:
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig("roc_curve.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("Saved ROC curve to roc_curve.png")
    except Exception as e:
        print("Could not plot ROC curve:", e)

if __name__ == "__main__":
    main()
