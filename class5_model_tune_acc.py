import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

DATA_PATH = "data/tourism_data_features.csv"
TARGET_COL = "ProdTaken"

def eval_one(model, X_train, X_test, y_train, y_test, threshold=0.5):
    model.fit(X_train, y_train)

    # probability -> threshold
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    acc = accuracy_score(y_test, pred)
    rec1 = recall_score(y_test, pred)  # recall for class 1
    return acc, rec1

def main():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Try multiple C values + both with/without class_weight
    Cs = [0.01, 0.1, 1, 3, 10, 30, 100]
    thresholds = [0.50, 0.55, 0.60]   # higher threshold often increases accuracy

    best = None

    for cw in [None, "balanced"]:
        for C in Cs:
            for th in thresholds:
                model = LogisticRegression(
                    max_iter=2000,
                    C=C,
                    class_weight=cw,
                    random_state=42
                )
                acc, rec1 = eval_one(model, X_train, X_test, y_train, y_test, threshold=th)

                print(f"class_weight={cw}, C={C}, threshold={th} -> acc={acc:.4f}, recall1={rec1:.4f}")

                # pick highest accuracy (tie-breaker: higher recall1)
                key = (acc, rec1)
                if best is None or key > (best["acc"], best["rec1"]):
                    best = {"cw": cw, "C": C, "th": th, "acc": acc, "rec1": rec1}

    print("\n=== BEST (by Accuracy) ===")
    print(best)

if __name__ == "__main__":
    main()
