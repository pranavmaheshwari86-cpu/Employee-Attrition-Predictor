import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "hr_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed"

# Create processed folder if it does not exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)


# 2. Remove unnecessary columns
columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(columns=columns_to_drop, inplace=True, errors="ignore")


# 3. Convert target variable into numerical values
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# 4. Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# 5. Encode categorical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)


# 6. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 7. Save processed datasets
X_train.to_csv(OUTPUT_PATH / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_PATH / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_PATH / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_PATH / "y_test.csv", index=False)


# 8. Display results
print("\nData Preprocessing Completed!")

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nProcessed files saved successfully!")
