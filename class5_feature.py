import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/tourism_data_cleaned.csv"
OUTPUT_PATH = "data/tourism_data_features.csv"
TARGET_COL = "ProdTaken"

def main():

    # ==============================
    # 1. LOAD CLEANED DATA
    # ==============================
    df = pd.read_csv(DATA_PATH)

    print("Loaded cleaned dataset:", df.shape)

    # ==============================
    # 2. SEPARATE TARGET
    # ==============================
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # ==============================
    # 3. ONE-HOT ENCODING
    # ==============================
    X = pd.get_dummies(X, drop_first=True)

    print("After one-hot encoding:", X.shape)

    # ==============================
    # 4. SCALING NUMERICAL FEATURES
    # ==============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Add target back
    X_scaled_df[TARGET_COL] = y.values

    # ==============================
    # 5. SAVE FEATURE DATASET
    # ==============================
    X_scaled_df.to_csv(OUTPUT_PATH, index=False)

    print("Feature dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
