import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


def train_model():
    df = pd.read_csv("data/churn.csv")

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # 🔥 SAVE EVERYTHING HERE (INSIDE FUNCTION)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    with open("model/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print("Model, encoders and columns saved successfully!")


if __name__ == "__main__":
    train_model()
