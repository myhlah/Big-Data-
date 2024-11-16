import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

# Sidebar for option selection
option = st.sidebar.radio("Web App", ['Classification Task', 'Regression Task'])
st.sidebar.write("""
    ### Guidelines:
    1. Upload your joblib file or dataset.
    2. Input data.
    3. Press the button to run the prediction model.
""")

# Function to load classification dataset
@st.cache_data
def load_data(uploaded_file):
    # Define column names as strings
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

def load_water_data(uploaded_file):
    names = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
            'chromium', 'copper', 'fluoride', 'bacteria', 'viruses', 'lead',
            'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium',
            'selenium', 'silver', 'uranium', 'is_safe']
    dataframe = pd.read_csv(uploaded_file, names=names)
    
    dataframe.replace('#NUM!', pd.NA, inplace=True)
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    # Impute missing values for numeric data (mean strategy)
    numeric_imputer = SimpleImputer(strategy='mean')
    dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

    # Encode target variable 'is_safe' if necessary (assuming it's categorical)
    if dataframe['is_safe'].dtype == 'object' or dataframe['is_safe'].isnull().any():
        label_encoder = LabelEncoder()
        dataframe['is_safe'] = label_encoder.fit_transform(dataframe['is_safe'].fillna(0))

    return dataframe

 
def get_water_quality_input():
    aluminium = st.number_input("Aluminium", min_value=0.000, value=0.000, key="aluminium1")
    ammonia = st.number_input("Ammonia", min_value=0.000, value=0.000, key="ammonia1")
    arsenic = st.number_input("Arsenic", min_value=0.000, value=0.000, key="arsenic1")
    barium = st.number_input("Barium", min_value=0.000, value=0.000, key="barium1")
    cadmium = st.number_input("Cadmium", min_value=0.000, value=0.000, key="cadmium1")
    chloramine = st.number_input("Chloramine", min_value=0.000, value=0.000, key="chloramine1")
    chromium = st.number_input("Chromium", min_value=0.000, value=0.000, key="chromium1")
    copper = st.number_input("Copper", min_value=0.000, value=0.000, key="copper1")
    fluoride = st.number_input("Fluoride", min_value=0.000, value=0.000, key="fluoride1")
    bacteria = st.number_input("Bacteria", min_value=0.000, value=0.000, key="bacteria1")
    viruses = st.number_input("Viruses", min_value=0.000, value=0.000, key="viruses1")
    lead = st.number_input("Lead", min_value=0.000, value=0.000, key="lead1")
    nitrates = st.number_input("Nitrates", min_value=0.000, value=0.000, key="nitrates1")
    nitrites = st.number_input("Nitrites", min_value=0.000, value=0.000, key="nitrites1")
    mercury = st.number_input("Mercury", min_value=0.000, value=0.000, key="mercury1")
    perchlorate = st.number_input("Perchlorate", min_value=0.000, value=0.000, key="perchlorate1")
    radium = st.number_input("Radium", min_value=0.000, value=0.000, key="radium1")
    selenium = st.number_input("Selenium", min_value=0.000, value=0.000, key="selenium1")
    silver = st.number_input("Silver", min_value=0.000, value=0.000, key="silver1")
    uranium = st.number_input("Uranium", min_value=0.000, value=0.000, key="uranium1")

    features = np.array([[aluminium, ammonia, arsenic, barium, cadmium, chloramine,
                        chromium, copper, fluoride, bacteria, viruses, lead,
                        nitrates, nitrites, mercury, perchlorate, radium,
                        selenium, silver, uranium]])
    return features


# Main function for the app
def main():
    if option == 'Classification Task':
        st.title("CLASSIFICATION TASK")
        
        # Upload model and predict
        st.subheader("Upload a Saved Model for Prediction")
        uploaded_file_classification = st.file_uploader("Upload Classification Model", key="classification_upload")

        if uploaded_file_classification is not None:
            model = joblib.load(uploaded_file_classification)  # Load the uploaded model

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

    elif option == 'Regression Task':
        st.title("REGRESSION TASK")
        # Model upload for prediction
        st.subheader("Upload a Saved Model for Prediction")
        uploaded_file_regression = st.file_uploader("Upload Regression Model", key="regression_upload")
        if uploaded_file_regression is not None:
            model = joblib.load(uploaded_file_regression)  # Load the uploaded model

            # Input fields for prediction
            st.subheader("Input Sample Data for Water Quality Prediction")
            input_data = get_water_quality_input()
            
            if st.button("Predict"):
                # Prediction
                prediction = model.predict(input_data)
                
                # Display prediction result
                st.subheader("Prediction Result and Interpretation")
                st.write(f"The predicted water safety is: **{'Safe' if prediction[0] == 1 else 'Not Safe'}**")
                interpretation = "The water is safe for human consumption." if prediction[0] == 1 else "The water is not safe for human consumption."
                st.write(interpretation)

if __name__ == "__main__":
    main()