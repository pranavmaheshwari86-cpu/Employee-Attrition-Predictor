# Employee Attrition Predictor

A Machine Learning-based application that predicts employee attrition using HR analytics data. The project implements three classification algorithms and provides an interactive Streamlit dashboard for employee-level predictions and model performance comparison.

## Project Overview

Employee attrition is a major challenge for organizations because employee turnover can increase recruitment costs, reduce productivity, and affect business performance.

This project uses historical HR data to identify patterns associated with employee attrition and predict whether an employee is likely to leave the organization.

## Key Features

- Employee attrition prediction using Machine Learning.
- Three selectable prediction models.
- Interactive dark-themed Streamlit dashboard.
- Model performance comparison.
- Confusion matrix visualization.
- ROC curve comparison.
- Prediction probability for individual employees.
- Exploratory Data Analysis (EDA).
- Data preprocessing and feature encoding.

## Machine Learning Algorithms

The project uses the following classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.17% | 34.52% | 61.70% | 44.27% | 79.83% |
| Decision Tree | 77.55% | 29.79% | 29.79% | 29.79% | 58.21% |
| Random Forest | 83.67% | 48.39% | 31.91% | 38.46% | 77.71% |

**Note:** Random Forest achieves the highest accuracy, while Logistic Regression provides the highest recall and F1 score. Model selection depends on the business objective.

## Dataset

- Dataset: IBM HR Analytics Employee Attrition Dataset
- Total records: 1,470
- Original features: 35
- Target variable: Attrition
- Target encoding: Yes = 1, No = 0
- Training/testing split: 80:20
- Encoded features: 44

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
Employee-Attrition-Predictor/
│
├── Employee-Attrition-Predictor/
│   ├── data/
│   │   ├── hr_data.csv
│   │   └── processed/
│   │
│   ├── models/
│   │   ├── logistic_regression.pkl
│   │   ├── decision_tree.pkl
│   │   ├── random_forest.pkl
│   │   ├── model_comparison.csv
│   │   └── evaluation/
│   │
│   ├── notebooks/
│   │
│   ├── src/
│   │   ├── eda.py
│   │   ├── preprocessing.py
│   │   ├── verify_preprocessing.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   │
│   ├── app.py
│   └── README.md
│
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/pranavmaheshwari86-cpu/Employee-Attrition-Predictor.git
```

### 2. Navigate to the project directory

```bash
cd Employee-Attrition-Predictor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run Employee-Attrition-Predictor/app.py
```

## Model Evaluation

The trained models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

Evaluation visualizations are stored in:

`Employee-Attrition-Predictor/models/evaluation/`

## Future Improvements

- Deploy the application online.
- Improve model performance through hyperparameter tuning.
- Add feature importance analysis.
- Integrate explainable AI techniques.
- Develop more advanced HR analytics visualizations.

## Author

**Pranav Maheshwari**

GitHub: [@pranavmaheshwari86-cpu](https://github.com/pranavmaheshwari86-cpu)

---

⭐ If you find this project interesting, consider giving it a star!
