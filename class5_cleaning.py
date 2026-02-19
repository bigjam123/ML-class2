import pandas as pd

# ==============================
# DATA CONFIG
# ==============================
DATA_PATH = "data/tourism_data.csv"
OUTPUT_PATH = "data/tourism_data_cleaned.csv"
TARGET_COL = "ProdTaken"


def main():

    # ==============================
    # 1. LOAD DATA
    # ==============================
    df = pd.read_csv(DATA_PATH)

    print("\n=== DATA LOADED ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ==============================
    # 2. CHECK MISSING VALUES
    # ==============================
    print("\n=== MISSING VALUES BEFORE CLEANING ===")
    print(df.isna().sum())

    # ==============================
    # 3. CHECK DUPLICATES
    # ==============================
    print("\n=== DUPLICATED ROWS ===")
    print("Duplicate rows:", df.duplicated().sum())

    # Remove duplicates
    df = df.drop_duplicates()

    # ==============================
    # 4. FIX INCONSISTENT CATEGORICAL VALUES
    # ==============================

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip()
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # ==============================
    # 5. HANDLE MISSING VALUES
    # ==============================

    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    print("\n=== MISSING VALUES AFTER CLEANING ===")
    print(df.isna().sum())

    # ==============================
    # 6. CHECK CLASS DISTRIBUTION
    # ==============================

    if TARGET_COL in df.columns:
        print("\n=== CLASS DISTRIBUTION ===")
        print(df[TARGET_COL].value_counts())
        print(df[TARGET_COL].value_counts(normalize=True))

    # ==============================
    # 7. SAVE CLEANED DATA
    # ==============================

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
