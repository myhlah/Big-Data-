import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Streamlit App

# Title and Description
st.title("Classification Task with K-Fold and Leave-One-Out Cross-Validation")

# Introduction and Instructions
st.write("""
### Instructions:
1. Upload your CSV dataset.
2. Select the target column (the variable you want to predict).
3. Select the feature columns (the variables used for prediction).
4. Click the "Run Cross-Validation" button to evaluate your models.
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

    if st.button("Run Cross-Validation"):
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

        # Create a Logistic Regression classifier
        model = LogisticRegression(max_iter=10000)

        # Model A: K-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        acc_kfold = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        log_loss_kfold = cross_val_score(model, X, y, cv=kf, scoring='neg_log_loss')

        # Model B: Leave-One-Out Cross-Validation (LOOCV)
        loo = LeaveOneOut()
        acc_loo = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
        log_loss_loo = cross_val_score(model, X, y, cv=loo, scoring='neg_log_loss')

        # Fit the model to get additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Confusion Matrix & Classification Report
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred)

        # Area Under ROC Curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = roc_auc_score(y, y_prob)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # Show results for Model A (K-Fold)
        st.write("### Model A (K-Fold Cross-Validation Results)")
        for i, score in enumerate(acc_kfold, 1):
            st.write(f"**Fold {i}:** {score:.2f}")

        mean_accuracy_kfold = acc_kfold.mean()
        std_deviation_kfold = acc_kfold.std()

        st.write(f"**Mean Accuracy (K-Fold):** {mean_accuracy_kfold:.2f}")
        st.write(f"**Standard Deviation of Accuracy (K-Fold):** {std_deviation_kfold:.2f}")

        # Show results for Model B (LOOCV)
        st.write("### Model B (Leave-One-Out Cross-Validation Results)")
        for i, score in enumerate(acc_loo, 1):
            st.write(f"**Fold {i}:** {score:.2f}")

        mean_accuracy_loo = acc_loo.mean()
        std_deviation_loo = acc_loo.std()

        st.write(f"**Mean Accuracy (LOOCV):** {mean_accuracy_loo:.2f}")
        st.write(f"**Standard Deviation of Accuracy (LOOCV):** {std_deviation_loo:.2f}")

        # Interpretation based on ranges
        accuracy_interpretation_kfold = ""
        stddev_interpretation_kfold = ""

        # Interpretation for K-Fold accuracy
        if mean_accuracy_kfold >= 0.90:
            accuracy_interpretation_kfold = "Excellent model performance."
        elif 0.80 <= mean_accuracy_kfold < 0.90:
            accuracy_interpretation_kfold = "Good model performance."
        elif 0.70 <= mean_accuracy_kfold < 0.80:
            accuracy_interpretation_kfold = "Fair model performance. Consider improving feature selection or model tuning."
        else:
            accuracy_interpretation_kfold = "Poor model performance. Needs significant improvement."

        # Interpretation for K-Fold standard deviation
        if std_deviation_kfold < 0.05:
            stddev_interpretation_kfold = "The model is very consistent across different folds."
        elif 0.05 <= std_deviation_kfold < 0.10:
            stddev_interpretation_kfold = "The model is fairly consistent, but there is some variation in performance."
        else:
            stddev_interpretation_kfold = "The model's performance is highly variable across different folds, indicating possible instability."

        # Displaying K-Fold interpretation
        st.write("### K-Fold Interpretation of Results")
        st.write(f"**Accuracy Interpretation (K-Fold):** {accuracy_interpretation_kfold}")
        st.write(f"**Standard Deviation Interpretation (K-Fold):** {stddev_interpretation_kfold}")

        # Interpretation for LOOCV
        accuracy_interpretation_loo = ""
        stddev_interpretation_loo = ""

        if mean_accuracy_loo >= 0.90:
            accuracy_interpretation_loo = "Excellent model performance."
        elif 0.80 <= mean_accuracy_loo < 0.90:
            accuracy_interpretation_loo = "Good model performance."
        elif 0.70 <= mean_accuracy_loo < 0.80:
            accuracy_interpretation_loo = "Fair model performance. Consider improving feature selection or model tuning."
        else:
            accuracy_interpretation_loo = "Poor model performance. Needs significant improvement."

        if std_deviation_loo < 0.05:
            stddev_interpretation_loo = "The model is very consistent across different folds."
        elif 0.05 <= std_deviation_loo < 0.10:
            stddev_interpretation_loo = "The model is fairly consistent, but there is some variation in performance."
        else:
            stddev_interpretation_loo = "The model's performance is highly variable across different folds, indicating possible instability."

        # Displaying LOOCV interpretation
        st.write("### Leave-One-Out Interpretation of Results")
        st.write(f"**Accuracy Interpretation (LOOCV):** {accuracy_interpretation_loo}")
        st.write(f"**Standard Deviation Interpretation (LOOCV):** {stddev_interpretation_loo}")

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
