###  1. Split into Train and Test Sets
import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # Import joblib to save and load the model
import numpy as np
import pandas as pd

# Set title for the app
st.title("Heart Disease Prediction Model")

# Introduction and Instructions
st.write("""
Part I - Classification Task        
### Instructions:
1. Upload your CSV dataset.
2. Click the button to evaluate the model.
""")


st.title("Split into Train and Test Sets")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file is not None:
     # Read dataset
    data = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview:")
    st.write(data.head())

    # Split into input and output variables
    array = data.values
    X = array[:, 0:13]
    Y = array[:, 13]
    st.write("### Shape of the Dataset:")
    st.write(data.shape)  # This will show you how many rows and columns are in your dataset

    # Set the test size using a slider
    test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
    seed = 7

    # Split the dataset into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Train the data on a Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)

    # Evaluate the accuracy
    result = model.score(X_test, Y_test)
    accuracy = result * 100
    st.write(f"Accuracy: {accuracy:.3f}%")

    # Interpretation based on accuracy ranges
    accuracy_interpretation = ""

    if accuracy >= 90:
        accuracy_interpretation = "Excellent performance! The model is highly accurate."
    elif 80 <= accuracy < 90:
        accuracy_interpretation = "Good performance. The model is performing well but can still be improved."
    elif 70 <= accuracy < 80:
        accuracy_interpretation = "Fair performance. The model might need some improvements."
    else:
        accuracy_interpretation = "Poor performance. Consider revisiting the model and features."

    # Display interpretation
    st.write(f"**Interpretation:** {accuracy_interpretation}")

    # Guide or Legend for easier understanding
    st.write("""
    ### Accuracy Guide:
    - **90% - 100%**: Excellent performance
    - **80% - 90%**: Good performance
    - **70% - 80%**: Fair performance
    - **Below 70%**: Poor performance, requires improvement
    """)
    

    # Train the model on the entire dataset and save it
    model.fit(X, Y)  # Train on the entire dataset
    model_filename = "logistic_regression_model.joblib"  # Filename to save the model
    joblib.dump(model, model_filename)  # Save the model to disk
    st.success(f"Model saved as {model_filename}")

    # Option to download the model
    with open(model_filename, "rb") as f:
        st.download_button("Download Model", f, file_name=model_filename)

    # Model upload for prediction
    st.subheader("Upload a Saved Model for Prediction")
    uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])



    if uploaded_model is not None:
        model = joblib.load(uploaded_model)  # Load the uploaded model

        # Sample input data for prediction
        st.subheader("Input Sample Data for Heart Disease Prediction")
        name = st.text_input("Enter your name:", "")
        age = st.number_input("Age", min_value=0, max_value=120, value=0)  # Age range 0-120
        sex = st.number_input("Sex (1 = male, 0 = female)", min_value=0, max_value=1, value=0)  # Sex is binary (0 or 1)
        cp = st.number_input("Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", min_value=0, max_value=3, value=0)  # 4 types of chest pain
        trestbps = st.number_input("Resting Blood Pressure (mm Hg) (80-200)", min_value=80, max_value=200, value=120)  # Typical range of resting BP
        chol = st.number_input("Serum Cholesterol (mg/dl) (100-600)", min_value=100, max_value=600, value=200)  # Cholesterol range
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", min_value=0, max_value=1, value=0)  # Fasting blood sugar is binary
        restecg = st.number_input("Resting Electrocardiographic Results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)", min_value=0, max_value=2, value=0)  # 3 types of ECG results
        thalach = st.number_input("Maximum Heart Rate Achieved (60-220)", min_value=60, max_value=220, value=150)  # Typical range for max heart rate
        exang = st.number_input("Exercise Induced Angina (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)  # Binary for exercise-induced angina
        oldpeak = st.number_input("ST depression induced by exercise relative to rest (0.0-7.0)", min_value=0.0, max_value=7.0, value=0.0)  # ST depression value
        slope = st.number_input("Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)", min_value=0, max_value=2, value=0)  # Slope categories
        ca = st.number_input("Number of major vessels (0-3) colored by fluoroscopy", min_value=0, max_value=3, value=0)  # Major vessels (0-3)
        thal = st.number_input("Thal (0 = normal; 1 = fixed defect; 2 = reversable defect)", min_value=0, max_value=2, value=0)
        
        # Note: 'target' is not used in the input_data for prediction
        target = st.number_input("Presence of heart disease in the patient (0 = no disease and 1 = disease)", min_value=0, max_value=1, value=0)

        # Creating input data array for prediction (ensure correct number of features)
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if st.button("Predict"):

            # Display the input data summary
            st.subheader("Input Data Summary")
            input_summary = {
                "Name": name if name else "Not Provided",
                "Age": age,
                "Sex": "Male" if sex == 1 else "Female",
                "Chest Pain Type": cp,
                "Resting Blood Pressure": trestbps,
                "Serum Cholesterol": chol,
                "Fasting Blood Sugar > 120 mg/dl": "Yes" if fbs == 1 else "No",
                "Resting ECG Results": restecg,
                "Max Heart Rate": thalach,
                "Exercise Induced Angina": "Yes" if exang == 1 else "No",
                "Oldpeak": oldpeak,
                "Slope of ST Segment": slope,
                "Major Vessels": ca,
                "Thal": thal,
                "Target": "Heart Disease" if target == 1 else "No Heart Disease"
            }
            st.write(input_summary)

             #Prediction
            prediction = model.predict(input_data)
            st.subheader("Prediction Result")
            
            # Display prediction based on the model's output
            if prediction[0] == 0:
                st.write("The predicted result is: **No Heart Disease**")
            else:
                st.write("The predicted result is: **Heart Disease**")

            # Add condition to compare with the target variable
            if target == 1 and prediction[0] == 1:
                st.write("The patient has heart disease based on input data and model prediction.")# If the target indicates heart disease (1) and the prediction also indicates heart disease (1).
            elif target == 1 and prediction[0] == 0:
                st.write("The patient is predicted not to have heart disease, but the input indicates they have heart disease.") #If the target indicates heart disease (1) but the model predicts no heart disease (0).
            elif target == 0 and prediction[0] == 0:
                st.write("The patient is predicted not to have heart disease, which is consistent with the input.")#If the target indicates no heart disease (0) and the prediction also indicates no heart disease (0).
            else:
                st.write("The patient is predicted to have heart disease, which is inconsistent with the input indicating no disease.") # If the target indicates no heart disease (0) but the prediction indicates heart disease (1).


             # Interpretation of input data
            st.write("### Interpretation of Input Data")
            st.write("""
                - **Age**: The age of the patient, which can affect heart disease risk.
                - **Sex**: Gender can influence heart disease prevalence.
                - **Chest Pain Type**: Different types of chest pain indicate various heart issues.
                - **Resting Blood Pressure**: High blood pressure can lead to heart disease.
                - **Serum Cholesterol**: High cholesterol levels can increase heart disease risk.
                - **Fasting Blood Sugar**: Indicates potential diabetes, a risk factor for heart disease.
                - **Resting ECG Results**: Reflects heart electrical activity; abnormalities may indicate heart issues.
                - **Max Heart Rate**: The maximum heart rate achieved during exercise; low levels may indicate heart problems.
                - **Exercise Induced Angina**: Indicates whether exercise causes chest pain.
                - **Oldpeak**: Reflects ST depression during exercise; higher values may indicate heart disease.
                - **Slope of ST Segment**: Indicates heart response to exercise; variations can indicate issues.
                - **Major Vessels**: The number of vessels colored by fluoroscopy can indicate heart health.
                - **Thal**: Reflects heart health; different values indicate various defects.
            """)

           
####################################################################################################
####################################################################################################

### 2. K-fold Cross Validation
import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    # Define column names as strings
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

def main():
    st.title("K-fold Cross Validation")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)

        # Check dataset content
        st.write("Dataset:")
        st.write(dataframe)

        # Checking dataset shape
        st.write("Number of Data (row,column)", dataframe.shape)

        # Check data types
        #st.write("Data Types:")
        # st.write(dataframe.dtypes)

        # Convert columns to numeric, forcing errors to NaN
        for column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

        # Checking for NaN values
        #st.write("Checking for NaN values in the dataset...")
        #st.write(dataframe.isnull().sum())

        # Drop or fill NaN values
        dataframe.fillna(0, inplace=True)  # Filling NaN with 0, can also choose to drop
                
        # Check unique values in the target variable
        #st.write("Unique values in target variable:", dataframe['target'].unique())
        st.write("Target Variable Distribution:")
        st.write(dataframe['target'].value_counts())

        # Checking for constant features
        constant_features = (dataframe.nunique() == 1).sum()
        #st.write(f"{constant_features} constant features found.")

        # Handling NaN values if necessary
        if dataframe.isnull().values.any():
            st.write("Filling NaN values with 0...")
            dataframe.fillna(0, inplace=True)  # Fill NaN values

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:13]
        Y = array[:, 13]

        # Ensure X and Y have valid values
        if len(Y) == 0 or Y.max() < 1:  # Check if Y contains valid target classes
            st.write("Target variable does not contain valid classes.")
            return

        num_folds = st.slider("Select number of folds for KFold Cross Validation:", 2, 10, 5)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=None)

        model = LogisticRegression(max_iter=210)
        st.write("Evaluating the model...")
        results = cross_val_score(model, X, Y, cv=kfold)

         # Check if results contain NaN values
        if results.size == 0 or np.isnan(results).all():
            st.write("Cross-validation results contain NaN values.")
        else:
            st.subheader("Cross-Validation Results")
            st.write(f"Accuracy: {results.mean() * 100:.3f}%")
            st.write(f"Standard Deviation: {results.std() * 100:.3f}%")
            
            # Interpretation and guidance
            st.write("### Interpretation of Results")
            st.write("""
                - **Accuracy** represents the proportion of correctly predicted instances out of the total instances.
                - A higher accuracy indicates that the model performs well on the dataset.
                - **Standard Deviation** provides insight into the variability of the accuracy across the different folds:
                  - A lower standard deviation indicates consistent performance across folds.
                  - A higher standard deviation suggests that the model's performance varies significantly with different subsets of data.
            """)

            # Legend for accuracy interpretation
            st.write("### Legend for Accuracy Interpretation")
            st.write("""
                - **90% - 100%**: Excellent model performance.
                - **70% - 89%**: Good model performance; may need some improvements.
                - **50% - 69%**: Fair model performance; consider tuning the model or using more data.
                - **Below 50%**: Poor model performance; likely to be worse than random guessing.
            """)
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
####################################################################################################
####################################################################################################

### 3. Leave-One-Out Cross Validation (LOOCV)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("Leave-One-Out Cross Validation (LOOCV)")

    # File uploader for CSV with a unique key
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader_unique_key")

    if uploaded_file is not None:
        # Load dataset
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Convert all columns to numeric, forcing errors to NaN
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Check for NaN values in the dataset
        #if dataframe.isnull().values.any():
            #st.warning("The dataset contains NaN values. Filling NaN with 0...")
            #dataframe.fillna(0, inplace=True)  # Fill NaN values

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:13]
        Y = array[:, 13]

        # Check if Y has valid values
        if len(Y) == 0 or Y.max() < 1 or Y.min() < 0:  # Check if Y contains valid target classes
            st.write("Target variable does not contain valid classes.")
            return

        # Split dataset into Leave One Out Cross Validation
        loocv = LeaveOneOut()

        # Train the data on a Logistic Regression model
        model = LogisticRegression(max_iter=500)

        # Evaluate using LOOCV
        st.write("Evaluating the model...")
        results = cross_val_score(model, X, Y, cv=loocv)

        # Display results
        st.subheader("Leave One Out Cross-Validation Results")
        if results.size == 0 or np.isnan(results).all():
            st.write("Cross-validation results contain NaN values.")
        else:
            st.write(f"Accuracy: {results.mean() * 100:.3f}%")
            st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

            # Interpretation and guide
            st.write("### Interpretation of Results")
            st.write("The Leave-One-Out Cross-Validation results show the model's performance.")
            st.write("Accuracy represents the percentage of correctly predicted instances over the total instances.")
            st.write("Standard deviation provides insight into the variability of the model's performance.")
            st.write("A high accuracy with a low standard deviation indicates a reliable model, while a low accuracy suggests that the model may not generalize well to unseen data.")
            st.write("Ensure the dataset is balanced and has sufficient representation of each class for better predictive performance.")

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
