
# ============================================================
# PROJECT 104: EMPLOYEE ATTRITION PREDICTOR
# STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "hr_data.csv"

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== TOTAL MISSING VALUES ==========")
print(df.isnull().sum().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 3. ATTRITION DISTRIBUTION
# ------------------------------------------------------------

print("\n========== ATTRITION DISTRIBUTION ==========")
print(df["Attrition"].value_counts())

print("\n========== ATTRITION PERCENTAGE ==========")
print(df["Attrition"].value_counts(normalize=True) * 100)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 4. AGE VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    bins=20,
    multiple="stack"
)

plt.title("Age Distribution by Employee Attrition")
plt.xlabel("Employee Age")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5. MONTHLY INCOME VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Employee Attrition")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. JOB SATISFACTION VS ATTRITION
# ------------------------------------------------------------

job_satisfaction_rate = (
    df.groupby("JobSatisfaction")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="AttritionPercentage")
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=job_satisfaction_rate,
    x="JobSatisfaction",
    y="AttritionPercentage"
)

plt.title("Job Satisfaction vs Attrition Percentage")
plt.xlabel("Job Satisfaction Level (1 = Low, 4 = High)")
plt.ylabel("Attrition Percentage (%)")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. WORK-LIFE BALANCE VS ATTRITION
# ------------------------------------------------------------

work_life_rate = (
    df.groupby("WorkLifeBalance")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="AttritionPercentage")
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=work_life_rate,
    x="WorkLifeBalance",
    y="AttritionPercentage"
)

plt.title("Work-Life Balance vs Attrition Percentage")
plt.xlabel("Work-Life Balance Level (1 = Low, 4 = High)")
plt.ylabel("Attrition Percentage (%)")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. OVERTIME VS ATTRITION
# ------------------------------------------------------------

overtime_rate = (
    df.groupby("OverTime")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="AttritionPercentage")
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=overtime_rate,
    x="OverTime",
    y="AttritionPercentage"
)

plt.title("Overtime vs Attrition Percentage")
plt.xlabel("Overtime")
plt.ylabel("Attrition Percentage (%)")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. YEARS AT COMPANY VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany"
)

plt.title("Years at Company vs Employee Attrition")
plt.xlabel("Employee Attrition")
plt.ylabel("Years at Company")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. DEPARTMENT-WISE ATTRITION
# ------------------------------------------------------------

department_rate = (
    df.groupby("Department")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="AttritionPercentage")
)

print("\n========== DEPARTMENT-WISE ATTRITION ==========")
print(department_rate)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=department_rate,
    x="Department",
    y="AttritionPercentage"
)

plt.title("Department-Wise Attrition Percentage")
plt.xlabel("Department")
plt.ylabel("Attrition Percentage (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. JOB ROLE-WISE ATTRITION
# ------------------------------------------------------------

job_role_rate = (
    df.groupby("JobRole")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="AttritionPercentage")
)

print("\n========== JOB ROLE-WISE ATTRITION ==========")
print(job_role_rate)

plt.figure(figsize=(11, 5))

sns.barplot(
    data=job_role_rate,
    x="JobRole",
    y="AttritionPercentage"
)

plt.title("Job Role-Wise Attrition Percentage")
plt.xlabel("Job Role")
plt.ylabel("Attrition Percentage (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------------------------

correlation_df = df.copy()

# Convert Attrition into numeric values for correlation
correlation_df["Attrition"] = (
    correlation_df["Attrition"].map({"Yes": 1, "No": 0})
)

# Select numerical columns only
numeric_df = correlation_df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(18, 12))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Employee Dataset")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13. FINAL EDA SUMMARY
# ------------------------------------------------------------

print("\n========== EDA COMPLETED SUCCESSFULLY ==========")

print("Total Employees:", len(df))

print(
    "Employees Who Left:",
    (df["Attrition"] == "Yes").sum()
)

print(
    "Employees Who Stayed:",
    (df["Attrition"] == "No").sum()
)

print(
    "Overall Attrition Percentage:",
    round((df["Attrition"] == "Yes").mean() * 100, 2),
    "%"
)

print("\nEDA analysis completed. Ready for preprocessing.")
