
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "hr_data.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #111318;
    color: #F5F5F5;
}

[data-testid="stSidebar"] {
    background-color: #191C23;
}

h1, h2, h3 {
    color: #F5F5F5;
}

p, label {
    color: #D1D5DB;
}

[data-testid="stMetric"] {
    background-color: #1C2029;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #303642;
}

div.stButton > button {
    background-color: #E63946;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #C92A37;
    color: white;
}

div[data-testid="stFormSubmitButton"] button {
    background-color: #E63946;
    color: white;
    width: 100%;
    border-radius: 8px;
    font-weight: bold;
}

hr {
    border-color: #303642;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD DATA AND MODELS
# ==========================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():

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

    return models


@st.cache_data
def load_model_results():

    return pd.read_csv(
        MODEL_PATH / "model_comparison.csv"
    )


try:

    df = load_dataset()

    models = load_models()

    model_results = load_model_results()

    X_train = pd.read_csv(
        PROCESSED_PATH / "X_train.csv"
    )

except Exception as e:

    st.error(f"Error loading project files: {e}")

    st.stop()


# ==========================================
# 5. PREPARE EMPLOYEE FEATURES
# ==========================================

columns_to_drop = [
    "Attrition",
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

feature_data = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

feature_columns = X_train.columns.tolist()


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("ATTRITION PREDICTOR")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Employee Prediction",
        "Model Comparison"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Employee Attrition Prediction System"
)

st.sidebar.caption(
    "Machine Learning Project"
)


# ==========================================
# 7. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.title("Employee Attrition Dashboard")

    st.markdown(
        "### HR Analytics and Machine Learning Prediction System"
    )

    st.markdown("---")

    total_employees = len(df)

    employees_left = (
        df["Attrition"] == "Yes"
    ).sum()

    employees_stayed = (
        df["Attrition"] == "No"
    ).sum()

    attrition_rate = (
        employees_left / total_employees
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Employees",
        total_employees
    )

    col2.metric(
        "Employees Left",
        employees_left
    )

    col3.metric(
        "Employees Stayed",
        employees_stayed
    )

    col4.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    st.markdown("---")

    st.subheader("Employee Attrition Overview")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("#### Attrition Distribution")

        attrition_counts = df["Attrition"].value_counts()

        st.bar_chart(attrition_counts)

    with chart_col2:

        st.markdown("#### Attrition Percentage")

        attrition_percentage = pd.DataFrame({
            "Category": ["Left", "Stayed"],
            "Percentage": [
                attrition_rate,
                100 - attrition_rate
            ]
        })

        st.bar_chart(
            attrition_percentage.set_index("Category")
        )

    st.markdown("---")

    st.subheader("About This Project")

    st.write(
        """
        Employee Attrition Predictor is a Machine Learning
        application designed to predict whether an employee
        is likely to leave an organization.

        The system uses three Machine Learning algorithms:

        - Logistic Regression
        - Decision Tree
        - Random Forest

        HR users can enter employee information and obtain
        predictions using their selected model.
        """
    )


# ==========================================
# 8. EMPLOYEE PREDICTION PAGE
# ==========================================

elif page == "Employee Prediction":

    st.title("Employee Attrition Prediction")

    st.write(
        "Enter employee information to predict attrition."
    )

    st.markdown("---")

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    st.markdown("---")

    st.subheader("Employee Information")

    input_data = {}

    with st.form("employee_prediction_form"):

        st.markdown(
            "### Enter Employee Details"
        )

        columns = st.columns(2)

        for index, column_name in enumerate(feature_data.columns):

            with columns[index % 2]:

                display_name = (
                    column_name.replace("_", " ")
                )

                if pd.api.types.is_numeric_dtype(
                    feature_data[column_name]
                ):

                    minimum = int(
                        feature_data[column_name].min()
                    )

                    maximum = int(
                        feature_data[column_name].max()
                    )

                    default = int(
                        feature_data[column_name].median()
                    )

                    input_data[column_name] = st.number_input(
                        display_name,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1
                    )

                else:

                    options = sorted(
                        feature_data[column_name]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    input_data[column_name] = st.selectbox(
                        display_name,
                        options
                    )

        st.markdown("---")

        submit_button = st.form_submit_button(
            "Predict Employee Attrition"
        )


    # ======================================
    # 9. MAKE PREDICTION
    # ======================================

    if submit_button:

        try:

            # Convert input into DataFrame

            input_df = pd.DataFrame(
                [input_data]
            )

            # Encode categorical variables

            input_encoded = pd.get_dummies(
                input_df,
                dtype=int
            )

            # Match training columns

            input_encoded = input_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Select model

            selected_model = models[
                selected_model_name
            ]

            # Predict

            prediction = selected_model.predict(
                input_encoded
            )[0]

            # Get prediction probability

            probabilities = selected_model.predict_proba(
                input_encoded
            )[0]

            classes = list(selected_model.classes_)

            attrition_probability = probabilities[
                classes.index(1)
            ] * 100

            # Display result

            st.markdown("---")

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            if prediction == 1:

                with result_col1:

                    st.error(
                        "Prediction: Employee Likely to Leave"
                    )

            else:

                with result_col1:

                    st.success(
                        "Prediction: Employee Likely to Stay"
                    )

            with result_col2:

                st.metric(
                    "Predicted Attrition Probability",
                    f"{attrition_probability:.2f}%"
                )

            st.progress(
                float(attrition_probability / 100)
            )

            st.info(
                f"Selected Model: {selected_model_name}"
            )

            st.caption(
                "Probability is the model's estimated score "
                "for the Attrition class, not a guarantee "
                "of the employee's future behavior."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ==========================================
# 10. MODEL COMPARISON PAGE
# ==========================================

elif page == "Model Comparison":

    st.title("Machine Learning Model Comparison")

    st.write(
        "Compare the performance of all three trained models."
    )

    st.markdown("---")

    st.dataframe(
        model_results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC AUC": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Performance Comparison")

    chart_data = model_results.set_index("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ]
    ]

    st.bar_chart(chart_data)

    best_model = model_results.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]

    st.markdown("---")

    st.subheader("Best Performing Model")

    st.success(
        f"Best Model: {best_model['Model']}"
    )

    st.metric(
        "Best F1 Score",
        f"{best_model['F1 Score']:.2%}"
    )


# ==========================================
# 11. FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Employee Attrition Predictor | "
    "Machine Learning Project"
)
