
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"

# Load processed datasets
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv")
y_test = pd.read_csv(DATA_PATH / "y_test.csv")


# 1. Check dataset shapes
print("========== DATASET SHAPES ==========")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# 2. Check missing values
print("\n========== MISSING VALUES ==========")

print("X_train missing values:", X_train.isnull().sum().sum())
print("X_test missing values:", X_test.isnull().sum().sum())
print("y_train missing values:", y_train.isnull().sum().sum())
print("y_test missing values:", y_test.isnull().sum().sum())


# 3. Check target values
print("\n========== TARGET VALUES ==========")

print("Training target:")
print(y_train["Attrition"].value_counts())

print("\nTesting target:")
print(y_test["Attrition"].value_counts())


# 4. Check column consistency
print("\n========== COLUMN CHECK ==========")

print("Training and testing columns match:",
      list(X_train.columns) == list(X_test.columns))


# 5. Check data types
print("\n========== DATA TYPES ==========")

print(X_train.dtypes.value_counts())


# 6. Check duplicate rows
print("\n========== DUPLICATE CHECK ==========")

print("Training duplicates:", X_train.duplicated().sum())
print("Testing duplicates:", X_test.duplicated().sum())


# 7. Final status
print("\n========== FINAL STATUS ==========")

if (
    X_train.isnull().sum().sum() == 0
    and X_test.isnull().sum().sum() == 0
    and y_train.isnull().sum().sum() == 0
    and y_test.isnull().sum().sum() == 0
    and list(X_train.columns) == list(X_test.columns)
    and set(y_train["Attrition"].unique()).issubset({0, 1})
    and set(y_test["Attrition"].unique()).issubset({0, 1})
):
    print("SUCCESS: Dataset is ready for model training!")
else:
    print("WARNING: Some preprocessing issues were found!")
