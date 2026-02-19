import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

DATA_PATH = "data/tourism_data_features.csv"
TARGET_COL = "ProdTaken"

def main():

    # ==============================
    # 1. LOAD DATA
    # ==============================
    df = pd.read_csv(DATA_PATH)
    print("Feature dataset loaded:", df.shape)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # ==============================
    # 2. TRAIN-TEST SPLIT
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # ==============================
    # 3. TRAIN MODEL
    # ==============================
    model = LogisticRegression(
        max_iter=2000,
        C=100,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # ==============================
    # 4. PREDICTION (Threshold 0.60)
    # ==============================
    threshold = 0.60
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ==============================
    # 5. EVALUATION
    # ==============================
    acc = accuracy_score(y_test, y_pred)
    rec1 = recall_score(y_test, y_pred)

    print("\n=== MODEL PERFORMANCE (Tuned) ===")
    print("Settings: class_weight=balanced, C=100, threshold=0.60")
    print("Accuracy:", round(acc, 4))
    print("Recall (class 1):", round(rec1, 4))

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix (raw values):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ==============================
    # 6. SAVE & PLOT CONFUSION MATRIX
    # ==============================

    # Create output folder if not exists
    os.makedirs("output", exist_ok=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Buyer (0)', 'Buyer (1)'],
        yticklabels=['Non-Buyer (0)', 'Buyer (1)']
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Tourism Product Model")

    plt.tight_layout()
    plt.savefig("output/confusion_matrix_class5.png")
    plt.show()

    print("\nConfusion matrix saved to: output/confusion_matrix_class5.png")


if __name__ == "__main__":
    main()
