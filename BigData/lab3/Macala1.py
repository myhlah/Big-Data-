import streamlit as st
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score # sampling techniques
from sklearn.linear_model import LogisticRegression # ML algorithm
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score #performance metric
from sklearn.naive_bayes import GaussianNB #hyperparameter tuning

import joblib

# Load CSS styling
st.markdown(
    """
    <style>
    .box {
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar for option selection
option = st.sidebar.radio("Part I - Classification Task", ['Sampling Technique/s', 'Performance Metric/s','Hyperparameter Tuning'])
st.sidebar.write("""  
    ### Instructions:
    1. Upload your CSV dataset.
    2. Click the button to evaluate the model.
      """)

# Classification Task
if option == 'Sampling Technique/s':
    st.title("Heart Disease Prediction Model")

    # RESAMPLING TECHNIQUES
    st.header("SAMPLING TECHNIQUES")
    # Introduction and Instructions
    st.write("""      
    ### Instructions:
    1. Upload your CSV dataset.
    2. Click the button to evaluate the model.
    3. Download the model.
    4. Upload the saved model for prediction.
    5. Input data and click the predict button.
    6. See result below the Input Data Summary section                   
             
    """)
   
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        # Define column names as strings
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe
    st.write("<br><br>", unsafe_allow_html=True)

    def main():
        #Split into Train and Test Sets
        st.title("Split into Train and Test Sets")

        # File uploader for dataset
        uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

        if uploaded_file is not None:
            # Read dataset
            data = pd.read_csv(uploaded_file)
            
            #st.write("### Dataset Preview:")
            #st.write(data.head())
            # Check dataset content
            st.write("Dataset:")
            st.write(data)

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
            model_filename = "logistic_regression_split_classification.joblib"  # Filename to save the model
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

    if __name__ == "__main__":
        main()      
####################################################################################################
####################################################################################################

# Performance Metric
elif option == 'Performance Metric/s':
    st.title("PERFORMANCE METRICS")
    # Introduction and Instruction
    st.write("""       
    ### Instructions:
    1. Upload your CSV dataset.
    2. Click the button to evaluate the model.
    """)
    st.write("<br><br>", unsafe_allow_html=True)

    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe
    ### Classification 
    def main():
        st.title("Classification Accuracy (K-fold Cross Validation)")

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader_1")

        if uploaded_file is not None:
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Check for missing values
            if dataframe.isnull().values.any():
                st.write("The dataset contains missing values. Please clean your data.")
                return

            # Convert categorical features to numeric if necessary
            categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                le = LabelEncoder()
                dataframe[col] = le.fit_transform(dataframe[col])

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            array = dataframe.values
            X = array[:, 0:13]  # Features
            Y = array[:, 13]     # Target

            # Check if dataset is too small for the number of folds
            num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 20, 10)
            if len(dataframe) < num_folds:
                st.write("The dataset is too small for the number of folds selected.")
                return

            kfold = KFold(n_splits=num_folds, shuffle=True, random_state=None)
            model = LogisticRegression(max_iter=210)

            # Calculate classification accuracy
            scoring = 'accuracy'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

            # Check if results contain any NaN
            if results is None or len(results) == 0 or any(pd.isna(results)):
                st.write("Error in cross-validation results. Please check your dataset.")
            else:
                st.subheader("Cross-Validation Results")
                st.write(f"Mean Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                plt.figure(figsize=(10, 5))
                plt.boxplot(results)
                plt.title('K-Fold Cross-Validation Accuracy')
                plt.ylabel('Accuracy')
                plt.xticks([1], [f'{num_folds}-Fold'])
                st.pyplot(plt)

                # Interpretation and guide
                st.write("### Interpretation of Results")
                st.write("- K-Fold Cross-Validation results show the model's performance.")
                st.write("- Mean accuracy represents the average percentage of correctly predicted instances over the total instances.")
                st.write("- Standard deviation indicates the variability of the model's performance across different folds.")
                st.write("- A high mean accuracy with a low standard deviation suggests a reliable model, while a low mean accuracy may indicate that the model is not generalizing well to unseen data.")
                st.write("- If the mean accuracy is low, consider feature engineering, experimenting with different algorithms, or hyperparameter tuning.")

        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()

    ### Classification Accuracy (split train-test 75:25 split ratio)
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe
    st.write("<br><br>", unsafe_allow_html=True)

    # Classification Accuracy 
    def main():
        st.title("Classification Accuracy (split train-test 75:25 split ratio)")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Check for missing values
            if dataframe.isnull().values.any():
                st.write("The dataset contains missing values. Please clean your data.")
                return

            # Convert categorical features to numeric if necessary
            categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                le = LabelEncoder()
                dataframe[col] = le.fit_transform(dataframe[col])

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, 0:13]
            Y = array[:, 13]

            # Split the dataset into a 75:25 train-test split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)

            # Train the data on a Logistic Regression model
            model = LogisticRegression(max_iter=210)
            model.fit(X_train, Y_train)

            # Evaluate the model on the test set
            accuracy = model.score(X_test, Y_test)
            
            # Display results
            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {accuracy * 100:.3f}%")

            # Interpretation and guide
            st.write("### Interpretation of Results")

            # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)

            #  first column
            with col1:
                st.write("- The classification accuracy obtained from the 75:25 train-test split provides insight into the model's performance.")
                st.write("- Accuracy represents the percentage of correctly predicted instances in the test set over the total instances.")
                st.write("- A high accuracy (e.g., above 75%) indicates that the model is effectively predicting the target variable.")
            # second column
            with col2:
                st.write("If the accuracy is low, consider the following:")
                st.write("- **Feature Engineering**: Investigate whether additional features can improve model performance.")
                st.write("- **Model Selection**: Experiment with different algorithms, such as decision trees or random forests.")
                st.write("- **Hyperparameter Tuning**: Adjust model parameters to optimize performance.")
                st.write("Moreover, analyze the confusion matrix to gain insights into specific misclassifications.")

        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()
    st.write("<br><br>", unsafe_allow_html=True)

    ### Classification Report
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("Classification Report")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader5")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Check for missing values
            if dataframe.isnull().values.any():
                st.write("The dataset contains missing values. Please clean your data.")
                return

            # Display data types
            #st.write("Data types in the dataset:")
            #st.write(dataframe.dtypes)

            # Convert categorical features to numeric if necessary
            categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                le = LabelEncoder()
                dataframe[col] = le.fit_transform(dataframe[col])

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, 0:13]
            Y = array[:, 13]

            # Split the dataset into train and test
            test_size = 0.33
            seed = 7
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the data on a Logistic Regression model
            model = LogisticRegression(max_iter=180)
            model.fit(X_train, Y_train)

            # Get predictions
            predicted = model.predict(X_test)

            # Get classification report
            report = classification_report(Y_test, predicted, output_dict=True)
            st.subheader("Classification Report")
            
            # Optionally, display the report as a DataFrame
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df)
            # Interpretation of the classification report
            st.subheader("Interpretation of the Classification Report")
            st.write("""
            The classification report provides several key metrics that help evaluate the performance of the model:
            - **Precision**: This indicates the proportion of true positive predictions made by the model out of all positive predictions. A higher precision means fewer false positives.
            
            - **Recall (Sensitivity)**: This metric represents the proportion of actual positive cases that were correctly identified by the model. A higher recall means the model is better at capturing positive cases and has fewer false negatives.
            
            - **F1 Score**: The F1 score is the harmonic mean of precision and recall. It provides a balance between the two metrics, and a higher F1 score indicates a better model performance.
            
            - **Support**: This shows the number of actual occurrences of the class in the specified dataset. It provides context to the precision and recall metrics, indicating how many samples of each class were present in the test set.

            To interpret the results, look for:
            - A balance between precision and recall, aiming for high values in both.
            - An F1 score closer to 1 is desirable, indicating a well-performing model.
            - Analyze support to understand the distribution of classes and if there are any imbalances that could affect the model's performance.
            """)

        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()

####################################################################################################
####################################################################################################

# Hyperparameter Tuning
elif option == 'Hyperparameter Tuning':
    st.title("HYPERPARAMETER TUNING")
    # Introduction and Instructions
    st.write("""       
        ### Instructions for Hyperparameter Tuning
        1. **Upload Dataset**  
             - Ensure heart.csv is available at the specified path.

        2. **Set Hyperparameters**  
             - Adjust **Test Size** (fraction of data for testing).
             - Set **Random Seed** for reproducibility.
             - Choose **Var Smoothing** (log scale) to adjust model smoothing.

        3. **Run Tuning**  
            - The model will train and display accuracy on the test set.

        4. **Result/s**  
            - See table below summarizes the results of each tuned model.
    """)

    st.write("<br><br>", unsafe_allow_html=True)

    ### Hyperparameter Tuning  
    st.header("Gaussian Naive Bayes")

    # Hyperparameter tuning function
    def main():
        # Load the dataset
        filename = 'E:\Myla docs\programs react\ITD105\lab3\heart.csv'
        dataframe = pd.read_csv(filename)
        array = dataframe.values
        X = array[:, 0:13]
        Y = array[:, 13]

        # Hyperparameter input fields in the main content area
        st.subheader("Set Hyperparameters")
        
        # Test size slider
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
        
        # Random seed slider
        random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
        
        # Var Smoothing range input (log scale)
        var_smoothing_range = st.slider("Var Smoothing Range (Log Scale)", -15, -1, (-9, -5), key="var_smoothing_range")

        # Button to run tuning
        if st.button("Run Tuning"):
            results = []  # Store results of each run

            # Loop over different var_smoothing values within selected range
            for log_smoothing in range(var_smoothing_range[0], var_smoothing_range[1] + 1):
                var_smoothing_value = 10 ** log_smoothing
                
                # Split the dataset
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                # Initialize and train the model
                model = GaussianNB(var_smoothing=var_smoothing_value)
                model.fit(X_train, Y_train)
                
                # Evaluate the model
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)

                # Save the results
                results.append({
                    "Test Size": test_size,
                    "Random Seed": random_seed,
                    "Var Smoothing (10^x)": f"10^{log_smoothing}",
                    "Accuracy (%)": f"{accuracy * 100:.2f}"
                })

            # Display results in a table
            results_df = pd.DataFrame(results)
            st.write("<br>", unsafe_allow_html=True)
            st.write("### Tuning Results")
            st.dataframe(results_df)

            # Display summary message
            st.write(f"Completed tuning over a range of var_smoothing values from 10^{var_smoothing_range[0]} to 10^{var_smoothing_range[1]}.")
            st.write("<br><br>", unsafe_allow_html=True)
            # Interpretation and Guide for Results
            st.write("### Interpretation of Results")
            st.write("""
                The accuracy score provides insight into how well the model predicts heart disease presence based on the test data.

                - **Accuracy (%)**: Higher accuracy indicates better model performance. However, keep in mind that accuracy alone doesn’t account for false positives or false negatives, which may be important in medical diagnosis.

                - **Var Smoothing**: This parameter helps manage how much the model smoothens class boundaries. Smaller values focus on fitting closely to the data, which may be prone to overfitting, while larger values introduce more smoothing, which can help avoid overfitting but may reduce model accuracy.
                
                #### Guidelines for Tuning:
                - **Test Size**: A test size around 20% (0.2) is typical. Increasing it means testing on a larger portion of data, which may give a more stable estimate of model performance but leaves less data for training.
                - **Var Smoothing**: Try a wide range (e.g., from 10^-9 to 10^-5) to observe any impact on accuracy. Adjust according to observed patterns; if accuracy varies significantly with changes in smoothing, fine-tune within that range.
                
                
            """)
    # Call the main function to run the code
    if __name__ == "__main__":
        main()

 #macala normailah itd105 it4d lab 3 part 1
