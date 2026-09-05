import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("credit_card_fraud_10k.csv")

x = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

numerical_features = x.select_dtypes(include=["number"]).columns.tolist()
categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=8,
        class_weight="balanced"
    ))
])

pipeline.fit(x_train, y_train)

y_pred_proba = pipeline.predict_proba(x_test)[:, 1]
y_pred = (y_pred_proba >= 0.30).astype(int)

print("--- Model Evaluation Metrics ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)*100:.2f}%")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, r"C:\credit_card_fraud\artifacts.joblib")
print("Successfully saved complete pipeline artifact.")
