
# Employee Attrition Predictor

## Project Overview

Employee Attrition Predictor is a Machine Learning project that predicts whether an employee is likely to leave an organization.

The project uses employee-related information such as age, department, job role, monthly income, job satisfaction, work experience and other HR attributes to predict employee attrition.

The main objective is to help organizations identify employees who may leave and take preventive actions.

## Objectives

- Analyze employee HR data using Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Train multiple Machine Learning models.
- Evaluate and compare model performance.
- Predict employee attrition through an interactive Streamlit application.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Algorithms

The following algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Project Structure

```text
Employee-Attrition-Predictor/
│
├── data/
│   ├── hr_data.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── model_comparison.csv
│   └── evaluation/
│
├── notebooks/
│
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── verify_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── app.py
├── README.md
└── .gitignore
```

## Dataset

The project uses an IBM HR Analytics Employee Attrition dataset.

- Total Records: 1470
- Total Original Features: 35
- Target Variable: Attrition
- Target Classes: Yes and No

The target variable is converted into:

- 1 = Employee Left
- 0 = Employee Stayed

## Model Performance

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.17% | 34.52% | 61.70% | 44.27% | 79.83% |
| Decision Tree | 77.55% | 29.79% | 29.79% | 29.79% | 58.21% |
| Random Forest | 83.67% | 48.39% | 31.91% | 38.46% | 77.71% |

Logistic Regression achieved the highest F1 Score and Recall among the three models.

Random Forest achieved the highest accuracy.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Employee-Attrition-Predictor
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run Employee-Attrition-Predictor/app.py
```

## Application Features

- Employee attrition dashboard.
- Employee-level attrition prediction.
- Selection between three trained ML models.
- Model performance comparison.
- Attrition probability visualization.

## Evaluation Visualizations

The project generates:

- Confusion matrices for all three models.
- ROC curve comparison.
- Model performance comparison chart.

These visualizations are stored in:

`models/evaluation/`

## Future Improvements

- Hyperparameter tuning.
- Cross-validation.
- Feature importance analysis.
- Improved prediction performance.
- Deployment on Streamlit Community Cloud.

## Author

Developed as a Machine Learning mini project.
