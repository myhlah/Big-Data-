import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Streamlit App

# Title and Description
st.title("I - Classification Task with K-Fold Cross-Validation")

# Introduction and Instructions
st.write("""
### Instructions:
1. Upload your CSV dataset.
2. Select the target column (the variable you want to predict).
3. Select the feature columns (the variables used for prediction).
4. Click the "Run K-Fold Cross-Validation" button to evaluate your model.
""")

# Load the Dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file is not None:
    # Read dataset
    data = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview:")
    st.write(data.head())

    # Select features and target
    target_column = st.selectbox("Select the target column (the variable you want to predict):", data.columns)
    feature_columns = st.multiselect("Select the feature columns (the variables used to predict the target):", data.columns, default=[col for col in data.columns if col != target_column])

    if st.button("Run K-Fold Cross-Validation"):
        st.write("### Selected Features and Target")
        st.write(f"**Target column:** {target_column}")
        st.write(f"**Feature columns:** {', '.join(feature_columns)}")

        X = data[feature_columns]
        y = data[target_column]

        # Convert non-numeric features to numeric (Label Encoding for categorical features)
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':  # If column is non-numeric
                if '/' in X[col].values[0]:  # Example to handle cases like '125/80'
                    X[col] = X[col].apply(lambda x: float(x.split('/')[0]))  # Use first number, modify as needed
                else:
                    # Use LabelEncoder for categorical features
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le

        # Also encode the target if it's categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        # Create a classifier
        clf = LogisticRegression()


        # K-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')

        # Show results with labels and explanation
        st.write("### K-Fold Cross-Validation Results")

        st.write("""
        **K-Fold Cross-Validation** divides your data into 5 folds (or subsets). It trains the model on 4 subsets and tests it on the remaining subset. This process is repeated 5 times with different folds. Below are the accuracy scores for each fold:
        """)

        for i, score in enumerate(cv_scores, 1):
            st.write(f"**Fold {i}:** {score:.2f}")

        # Displaying mean accuracy and standard deviation
        mean_accuracy = cv_scores.mean()
        std_deviation = cv_scores.std()

        st.write(f"**Mean Accuracy (across all folds):** {mean_accuracy:.2f}")
        st.write(f"**Standard Deviation of Accuracy (how much the scores vary):** {std_deviation:.2f}")

        # Interpretation based on ranges
        accuracy_interpretation = ""
        stddev_interpretation = ""

        # Interpretation for accuracy
        if mean_accuracy >= 0.90:
            accuracy_interpretation = "Excellent model performance."
        elif 0.80 <= mean_accuracy < 0.90:
            accuracy_interpretation = "Good model performance."
        elif 0.70 <= mean_accuracy < 0.80:
            accuracy_interpretation = "Fair model performance. Consider improving feature selection or model tuning."
        else:
            accuracy_interpretation = "Poor model performance. Needs significant improvement."

        # Interpretation for standard deviation
        if std_deviation < 0.05:
            stddev_interpretation = "The model is very consistent across different folds."
        elif 0.05 <= std_deviation < 0.10:
            stddev_interpretation = "The model is fairly consistent, but there is some variation in performance."
        else:
            stddev_interpretation = "The model's performance is highly variable across different folds, indicating possible instability."

        # Displaying the interpretation
        st.write("### Interpretation of Results")
        st.write(f"**Accuracy Interpretation:** {accuracy_interpretation}")
        st.write(f"**Standard Deviation Interpretation:** {stddev_interpretation}")

        st.write("""
        **General Guide:**
        - **Accuracy**:
          - **0.90 - 1.00**: Excellent performance.
          - **0.80 - 0.90**: Good performance.
          - **0.70 - 0.80**: Fair performance.
          - **Below 0.70**: Poor performance.
        
        - **Standard Deviation**:
          - **Below 0.05**: Very consistent performance.
          - **0.05 - 0.10**: Fairly consistent performance.
          - **Above 0.10**: Inconsistent performance, model may be unstable.
        """)
