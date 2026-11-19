
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)


# STEP 1: Project paths

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"

OUTPUT_PATH = MODEL_PATH / "evaluation"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# STEP 2: Load testing data

X_test = pd.read_csv(DATA_PATH / "X_test.csv")
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze("columns")

print("Testing dataset loaded successfully!")


# STEP 3: Load trained models

models = {
    "Logistic Regression": joblib.load(
        MODEL_PATH / "logistic_regression.pkl"
    ),

    "Decision Tree": joblib.load(
        MODEL_PATH / "decision_tree.pkl"
    ),

    "Random Forest": joblib.load(
        MODEL_PATH / "random_forest.pkl"
    )
}


# STEP 4: Generate confusion matrices

for name, model in models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Attrition", "Attrition"]
    )

    display.plot()

    plt.title(name + " - Confusion Matrix")

    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_PATH / f"{filename}_confusion_matrix.png"
    )

    plt.close()

    print(name, "confusion matrix saved!")


# STEP 5: Generate ROC curve

plt.figure(figsize=(8, 6))

ax = plt.gca()

for name, model in models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_PATH / "roc_curve_comparison.png")

plt.close()

print("ROC curve saved!")


# STEP 6: Model comparison graph

results = pd.read_csv(
    MODEL_PATH / "model_comparison.csv"
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Machine Learning Model Comparison")

plt.ylabel("Score")

plt.xlabel("Machine Learning Models")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend(loc="best")

plt.tight_layout()

plt.savefig(OUTPUT_PATH / "model_comparison.png")

plt.close()

print("Model comparison graph saved!")


# STEP 7: Display classification reports

for name, model in models.items():

    y_pred = model.predict(X_test)

    print("\n================================")
    print(name)
    print("================================")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


print("\n================================")
print("MODEL EVALUATION COMPLETED!")
print("================================")

print("All visualizations saved successfully!")
