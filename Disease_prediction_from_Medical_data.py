import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.datasets import load_breast_cancer

def train_and_evaluate(X, y, dataset_name):
    # Split dataset into 80% train, 20% test, stratified by label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print(f"\n==== Results for {dataset_name} dataset ====")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{model_name}:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        results[model_name] = acc
    return results

heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]
heart = pd.read_csv(heart_url, names=heart_cols, na_values="?")
heart = heart.dropna().reset_index(drop=True)
# Convert target to binary: 0 = no disease, 1 = disease (presence of heart disease)
heart['target'] = (heart['target'] > 0).astype(int)
X_heart = heart.drop('target', axis=1)
y_heart = heart['target']

diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'target'
]
diabetes = pd.read_csv(diabetes_url, names=diabetes_cols)
X_diabetes = diabetes.drop('target', axis=1)
y_diabetes = diabetes['target']

breast_data = load_breast_cancer()
X_breast = pd.DataFrame(breast_data.data, columns=breast_data.feature_names)
y_breast = pd.Series(breast_data.target)

heart_results = train_and_evaluate(X_heart, y_heart, 'Heart Disease')
diabetes_results = train_and_evaluate(X_diabetes, y_diabetes, 'Diabetes')
breast_results = train_and_evaluate(X_breast, y_breast, 'Breast Cancer')

results_df = pd.DataFrame({
    "Heart Disease": heart_results,
    "Diabetes": diabetes_results,
    "Breast Cancer": breast_results
})
results_df.plot(kind='bar', figsize=(10,6))
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.show()
     
