import streamlit as st
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut, ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, ConfusionMatrixDisplay,roc_auc_score, roc_curve
import joblib

# Sidebar for option selection
option = st.sidebar.radio("Streamlit App", ['Classification Task', 'Regression Task'])
st.sidebar.write("""  
    ### Instructions:
    1. Upload your CSV dataset.
    2. Click the button to evaluate the model.
      """)

# Function to load the dataset with caching
@st.cache_data
def load_data(uploaded_file):
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Classification Task
if option == 'Classification Task':
    st.title("Heart Disease Prediction Model")

    # RESAMPLING TECHNIQUES
    st.header("RESAMPLING TECHNIQUES")
    # Split into Train and Test Sets
    st.header("Split into Train and Test Sets")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

    if uploaded_file is not None:
        # Read dataset
        data = pd.read_csv(uploaded_file)
        
        # Create two columns for Data Info and Missing Values
        col1, col2 = st.columns(2)

        #  first column
        with col1:
            st.write("### Dataset Preview:")
            st.write(data.head())
            

        # second column
        with col2:
            # Split into input and output variables
            X = data.iloc[:, :-1].values  # All columns except the last one
            Y = data.iloc[:, -1].values    # Last column
            st.write("### Shape of the Dataset:")
            st.write(data.shape)

            # Set the test size using a slider
            test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
            seed = 7

            # Split the dataset into test and train
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the Logistic Regression model
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test) * 100
            st.write(f"Accuracy: {accuracy:.3f}%")

            # Interpretation based on accuracy ranges
            if accuracy >= 90:
                interpretation = "Excellent performance! The model is highly accurate."
            elif 80 <= accuracy < 90:
                interpretation = "Good performance. The model is performing well but can still be improved."
            elif 70 <= accuracy < 80:
                interpretation = "Fair performance. The model might need some improvements."
            else:
                interpretation = "Poor performance. Consider revisiting the model and features."
            
            st.write(f"**Interpretation:** {interpretation}")

        # Accuracy guide
        st.write("""
        ### Accuracy Guide:
        - **90% - 100%**: Excellent performance
        - **80% - 90%**: Good performance
        - **70% - 80%**: Fair performance
        - **Below 70%**: Poor performance, requires improvement
        """)

        # Train the model on the entire dataset and save it
        model.fit(X, Y)
        model_filename = "logistic_regression_model.joblib"
        joblib.dump(model, model_filename)
        st.success(f"Model saved as {model_filename}")

        # Option to download the model
        with open(model_filename, "rb") as f:
            st.download_button("Download Model", f, file_name=model_filename)

        # Model upload for prediction
        st.subheader("Upload a Saved Model for Prediction")
        uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])

        if uploaded_model is not None:
            model = joblib.load(uploaded_model)

            # Sample input data for prediction
            st.subheader("Input Sample Data for Heart Disease Prediction")
            name = st.text_input("Enter your name:", "")
            age = st.number_input("Age", min_value=0, max_value=120, value=0)
            sex = st.number_input("Sex (1 = male, 0 = female)", min_value=0, max_value=1, value=0)
            cp = st.number_input("Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", min_value=0, max_value=3, value=0)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg) (80-200)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Serum Cholesterol (mg/dl) (100-600)", min_value=100, max_value=600, value=200)
            fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", min_value=0, max_value=1, value=0)
            restecg = st.number_input("Resting Electrocardiographic Results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)", min_value=0, max_value=2, value=0)
            thalach = st.number_input("Maximum Heart Rate Achieved (60-220)", min_value=60, max_value=220, value=150)
            exang = st.number_input("Exercise Induced Angina (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)
            oldpeak = st.number_input("ST depression induced by exercise relative to rest (0.0-7.0)", min_value=0.0, max_value=7.0, value=0.0)
            slope = st.number_input("Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)", min_value=0, max_value=2, value=0)
            ca = st.number_input("Number of major vessels (0-3) colored by fluoroscopy", min_value=0, max_value=3, value=0)
            thal = st.number_input("Thal (0 = normal; 1 = fixed defect; 2 = reversable defect)", min_value=0, max_value=2, value=0)
            target = st.number_input("Presence of heart disease in the patient (0 = no disease and 1 = disease)", min_value=0, max_value=1, value=0)

        # Create two columns for Data Info and Missing Values
        col1, col2 = st.columns(2)

        #  first column
        with col1:
            # Creating input data array for prediction
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

                # Prediction
                prediction = model.predict(input_data)
                st.subheader("Prediction Result")
                
                if prediction[0] == 0:
                    st.write("The predicted result is: **No Heart Disease**")
                else:
                    st.write("The predicted result is: **Heart Disease**")

                # Comparison with the target variable
                if target == 1 and prediction[0] == 1:
                    st.write("The patient has heart disease based on input data and model prediction.")
                elif target == 1 and prediction[0] == 0:
                    st.write("The patient is predicted not to have heart disease, but the input indicates they have heart disease.")
                elif target == 0 and prediction[0] == 0:
                    st.write("The patient is predicted not to have heart disease, which is consistent with the input.")
                else:
                    st.write("The patient is predicted to have heart disease, while the input indicates otherwise.")
            

        # second column
        with col2:
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

            # Convert columns to numeric, forcing errors to NaN
            for column in dataframe.columns:
                dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

            # Drop or fill NaN values
            dataframe.fillna(0, inplace=True)  # Filling NaN with 0, can also choose to drop

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
                
                # Create two columns for Data Info and Missing Values
                col1, col2 = st.columns(2)

                #  first column
                with col1:
                    # Interpretation and guidance
                    st.write("### Interpretation of Results")
                    st.write("""
                        - **Accuracy** represents the proportion of correctly predicted instances out of the total instances.
                        - A higher accuracy indicates that the model performs well on the dataset.
                        - **Standard Deviation** provides insight into the variability of the accuracy across the different folds:
                        - A lower standard deviation indicates consistent performance across folds.
                        - A higher standard deviation suggests that the model's performance varies significantly with different subsets of data.
                    """)
                    
                # second column
                with col2:
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

            # Drop rows with NaN values
            dataframe.dropna(inplace=True)

            # Preparing the data
            array = dataframe.values
            X = array[:, 0:13]
            Y = array[:, 13]

            # Check if Y has valid values
            if len(Y) == 0 or Y.max() < 1 or Y.min() < 0:  # Check if Y contains valid target classes
                st.write("Target variable does not contain valid classes.")
                return

            # Scale the features to ensure numerical stability
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split dataset into Leave One Out Cross Validation
            loocv = LeaveOneOut()

            # Train the data on a Logistic Regression model
            model = LogisticRegression(max_iter=500)

            # Evaluate using LOOCV
            st.write("Evaluating the model...")
            results = cross_val_score(model, X_scaled, Y, cv=loocv)

            # Display results
            st.subheader("Leave One Out Cross-Validation Results")
            if results.size == 0 or np.isnan(results).all():
                st.write("Cross-validation results contain NaN values.")
            else:
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Interpretation and guide
                st.write("### Interpretation of Results")
                st.write("- The Leave-One-Out Cross-Validation results show the model's performance.")
                st.write("- Accuracy represents the percentage of correctly predicted instances over the total instances.")
                st.write("- Standard deviation provides insight into the variability of the model's performance.")
                st.write("- A high accuracy with a low standard deviation indicates a reliable model, while a low accuracy suggests that the model may not generalize well to unseen data.")
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()

    ### Performance Metrics Here
    st.title("PERFORMANCE METRICS")
    ### 1.1 Classification Accuracy (K-fold Cross Validation)

    st.header("CLASSIFICATION METRICS")

    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    def main():
        st.title("1.1 Classification Accuracy (K-fold Cross Validation)")

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

    ### 1.2 Classification Accuracy (split train-test 75:25 split ratio)
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("1.2 Classification Accuracy (split train-test 75:25 split ratio)")

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

    ### 2. Logarithmic Loss
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("2. Logarithmic Loss (Log Loss) using K-Fold")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader3")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Check for missing values
            if dataframe.isnull().values.any():
                st.write("The dataset contains missing values. Please clean your data.")
                return

            # Check the first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Convert columns to numeric and handle non-numeric values
            for col in dataframe.columns:
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

            # Drop rows with NaN values
            dataframe.dropna(inplace=True)

            # Check for valid classes in target variable
            if dataframe['target'].nunique() < 2:
                st.write("The target variable does not contain at least two unique classes. Please check your dataset.")
                return

            # Preparing the data
            array = dataframe.values
            X = array[:, 0:13]
            Y = array[:, 13]

            # Split the dataset into a 10-fold cross validation
            num_folds = 10
            kfold = KFold(n_splits=num_folds, random_state=None)

            # Train the data on a Logistic Regression model
            model = LogisticRegression(max_iter=200)

            # Calculate the log-loss for each fold
            scoring = 'neg_log_loss'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

            # Check for NaN values in results
            if np.isnan(results).any():
                st.write("Log loss calculation resulted in NaN values. This may occur due to invalid predictions. Please check your dataset.")
                return

            # Convert negative log loss to positive for better interpretation
            log_loss_values = -results

            # Display results
            st.subheader("Cross-Validation Results")
            st.write(f"Mean LogLoss: {log_loss_values.mean():.3f} (±{log_loss_values.std():.3f})")

            # Plotting LogLoss for each fold
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_folds + 1), log_loss_values, marker='o', linestyle='-')
            plt.title('LogLoss for Each Fold of Cross-Validation')
            plt.xlabel('Fold Number')
            plt.ylabel('LogLoss')
            plt.xticks(range(1, num_folds + 1))
            plt.grid()

            # Check for valid log loss values before plotting
            if log_loss_values.max() > 0:  # Ensure max log loss is valid
                plt.ylim(0, log_loss_values.max() + 0.5)  # Adjusting y-axis for better visualization
            else:
                plt.ylim(0, 1)  # Default limits if max log loss is not valid

            st.pyplot(plt)  # Display the plot in Streamlit

            # Interpretation and guide
            st.write("### Interpretation of Results")
            st.write("Logarithmic Loss (Log Loss) measures the performance of a classification model where the prediction is a probability value between 0 and 1.")
            st.write("- A lower log loss value indicates better performance of the model. It penalizes false classifications more severely than correct classifications.")
            st.write("- Mean Log Loss provides an average value of the log loss across all folds, and the standard deviation indicates the variability in performance.")
            st.write("If you observe high log loss values, consider adjusting the model or exploring different features.")
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()

    ### 3. Confusion Matrix
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("3. Confusion Matrix")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader4")

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

            # Ensure all features are numeric
            #st.write("Data types after encoding:")
            #st.write(dataframe.dtypes)

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
            model = LogisticRegression(max_iter=280)
            model.fit(X_train, Y_train)

            # Calculate predictions
            predicted = model.predict(X_test)

            # Calculate confusion matrix
            matrix = confusion_matrix(Y_test, predicted)

            # Plot the confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            st.pyplot(fig)  # Display the plot in Streamlit
            
            # Interpretation and guide
            st.write("### Interpretation of Results")

            # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)

            #  first column
            with col1:
                
                st.write("The confusion matrix provides a summary of the prediction results on a classification problem.")
                st.write("Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).")
                st.write("The diagonal elements represent the number of instances correctly classified, while the off-diagonal elements indicate misclassifications.")
                st.write("For example, if the matrix shows: ")
                st.write("```\n[[TP, FN],\n [FP, TN]]\n```")
                st.write("where: ")
                st.write("- **TP** (True Positives): Correctly predicted positive cases")
                st.write("- **TN** (True Negatives): Correctly predicted negative cases")
                st.write("- **FP** (False Positives): Incorrectly predicted as positive")
                st.write("- **FN** (False Negatives): Incorrectly predicted as negative")

            # second column
            with col2:
                st.write("From these values, you can calculate performance metrics such as accuracy, precision, recall, and F1 score to evaluate your model's effectiveness.")
                st.write("A well-performing model should have high true positives and true negatives while minimizing false positives and false negatives.")

        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()

    ### 4. Classification Report
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("4. Classification Report")

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

    ### 5. Area Under ROC Curve

    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("5. ROC AUC Score and Curve (75:25 Train-Test Split)")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader6")

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

            # Predict the probabilities for the test set
            Y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate the ROC AUC score
            roc_auc = roc_auc_score(Y_test, Y_prob)
            st.write("AUC: %.3f" % roc_auc)

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)

            # Plotting the ROC curve
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Interpretation and guide
            st.write("### Interpretation of Results")
            st.write("The ROC curve (Receiver Operating Characteristic curve) is a graphical representation of the performance of a binary or multiclass classifier system as its discrimination threshold is varied.")
            st.write("It plots the True Positive Rate (TPR) against the False Positive Rate (FPR).")
            
            # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)

            # First column
            with col1:
                st.write("**Key Terms:**")
                st.write("- **True Positive Rate (TPR)**: Also known as sensitivity or recall, it represents the proportion of actual positives that are correctly identified.")
                st.write("- **False Positive Rate (FPR)**: Represents the proportion of actual negatives that are incorrectly identified as positives.")
                st.write("The area under the ROC curve (AUC) is a single scalar value that summarizes the performance of the model.")
            
            # Second column
            with col2:
                st.write("**Interpretation of AUC values:**")
                st.write("- **AUC = 1.0**: Perfect model with no false positives or false negatives.")
                st.write("- **AUC = 0.5**: No discriminative ability, similar to random guessing.")
                st.write("- **AUC < 0.5**: The model performs worse than random guessing.")
                st.write("In general, higher AUC values indicate better model performance.")

        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()



# Regression Task
elif option == 'Regression Task':
    st.title("Climate Prediction Model")

    # RESAMPLING TECHNIQUES
    st.header("RESAMPLING TECHNIQUES")
    # Split into Train and Test Sets
    st.title("Split into Train and Test Sets")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv", key="file_uploader_unique_key")

    if uploaded_file is not None:
        # Read dataset
        data = pd.read_csv(uploaded_file)

        # Normalize column names
        data.columns = data.columns.str.strip().str.lower()  # Lowercase and strip whitespace

        st.write("### Dataset Preview:")
        st.write(data.head())  # Show the first few rows of the dataset

        st.write("### Correlation Matrix:")
        st.write(data.corr())

        # Check for any non-numeric columns
        non_numeric_cols = data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            st.write(f"Non-numeric columns found: {non_numeric_cols}. Please ensure only numeric data is used for model training.")

        # Ensure columns
        if len(data.columns) < 5:  # Change from 11 to 5
            st.error("The dataset must have at least 5 columns.")
        else:
            # Split into input and output variables
            array = data.values
            X = array[:, 0:9]  
            Y = array[:, 9]   

            st.write("### Shape of the Dataset:")
            st.write(data.shape)  # This will show you how many rows and columns are in your dataset

            # Set the test size using a slider
            test_size = st.slider("Test size (as a percentage)", 20, 100, 50) / 100
            seed = 7

            # Split the dataset into test and train
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the data on a Linear Regression model
            model = LinearRegression()
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            result = model.score(X_test, Y_test)
            accuracy = result * 100
            st.subheader("Split into Train and Test Sets Results")
            st.write(f"Accuracy: {accuracy:.3f}%")

            # Interpretation of the accuracy
            if accuracy < 0:
                st.warning("The model's performance is poor. Please check your data for quality and relevance.")
            elif accuracy < 50:
                st.warning("The model has low predictive power. Consider gathering more data or using different features.")
            elif accuracy < 75:
                st.success("The model has moderate predictive power. It may perform reasonably well.")
            else:
                st.success("The model has high predictive power. Good job!")

            # Save and download the model
            model.fit(X, Y)
            model_filename = "linear_regression_model.joblib"
            joblib.dump(model, model_filename)
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
        st.subheader("Input Sample Data for Climate Prediction")
        temp = st.number_input("Temperature (0 to 50 Celsius)", min_value=-50, max_value=50, value=0)  
        hum = st.number_input("Humidity (1-100%)", min_value=0, max_value=100, value=0)  
        wind = st.number_input("Wind Speed (0-20)", min_value=0, max_value=20, value=0)  
        pres = st.number_input("Pressure (1-100%)", min_value=1, max_value=100, value=30)  
        heat = st.number_input("Heat Index (0 to 70 Celsius)", min_value=0, max_value=70, value=0)  
        dew = st.number_input("Dew Point (0 to 30 Celsius)", min_value=0, max_value=30, value=0)  
        chill = st.number_input("Wind Chill Index (0-100)", min_value=0, max_value=100, value=0)  
        temphum = st.number_input("Comfort Level (0-Normal, 1-Fair, 2-Uncomfortable, 3-Hot, 4-Cold )", min_value=0, max_value=4, value=0)  
        humwind = st.number_input("Air Level(0-Calm, 1-Breezy, 2-Windy, 3-Strong Wind, 4-Gale )", min_value=0, max_value=4, value=0)
        events = st.number_input("Weather Event (1-Event Occurred or 0-No Event)", min_value=0, max_value=1, value=0)  

        # Creating input data array for prediction (ensure correct number of features)
        input_data = np.array([[temp, hum, wind, pres, heat, dew, chill, temphum, humwind]])

        if st.button("Predict"):
            # Display the input data summary
            st.subheader("Input Data Summary")
            input_summary = {
                "Temperature (0 to 50 Celsius)": temp,
                "Humidity (1-100%)": hum,
                "Wind Speed (0-20)": wind,
                "Pressure Pressure (1-100%)": pres,
                "Heat Index (0 to 70 Celsius)": heat,
                "Dew Point (1-100%)": dew,
                "Wind Chill Index (0-100)": chill,
                "Comfort Level (0-Normal, 1-Fair, 2-Uncomfortable, 3-Hot, 4-Cold )": temphum,
                "Air Level(0-Calm, 1-Breezy, 2-Windy, 3-Strong Wind, 4-Gale )": humwind,
                "Weather Event (1-Event Occurred or 0-No Event)": events,
            }
            st.write(input_summary)

            # Prediction
            prediction = model.predict(input_data)
            st.subheader("Prediction Result")
            
            # Display prediction based on the model's output
            st.write(f"The predicted climate metric is: **{prediction[0]:.3f}**")

            # Interpretation based on prediction value
            if prediction[0] < 0:
                interpretation = "The climate conditions seem stable. No immediate climate anomalies predicted."
            elif 0 <= prediction[0] < 10:
                interpretation = "There might be slight climate variations or mild changes. Monitor conditions regularly."
            elif 10 <= prediction[0] < 20:
                interpretation = "Moderate climate variations predicted. Potential for changes in weather patterns or events."
            elif 20 <= prediction[0] < 30:
                interpretation = "Significant climate changes predicted. Possible risk of extreme weather events. Stay alert."
            else:
                interpretation = "Severe climate anomalies predicted. High chance of extreme weather conditions or environmental impacts."

            st.write(interpretation)

            st.write("""
            ### Prediction Guide:
            - **Below 0**: Climate conditions are stable, no notable changes.
            - **0 to 10**: Mild climate changes, typically not alarming.
            - **10 to 20**: Moderate changes, potential for weather pattern shifts or unusual events.
            - **20 to 30**: Significant changes, likely to experience abnormal weather patterns or environmental stress.
            - **Above 30**: Severe changes, strong potential for extreme weather events or serious environmental impacts.
            """)

    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['temp', 'hum', 'wind', 'pres', 'heat', 'dew', 'chill', 'temphum', 'humwind', 'target']  # Ensure correct column names
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    # Main app
    def main():
        st.title("Repeated Random Test-Train Splits ")

        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader_shuffle")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            if dataframe.isnull().values.any():
                st.write("Missing values detected! Filling missing values with column means.")
                #dataframe = dataframe.fillna(dataframe.mean())  # Fills missing values with column mean

            # Ensure all columns are numeric
            dataframe = dataframe.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric data to NaN and then handle missing
            #st.write("Converting non-numeric columns to numeric...")

            # Check again for any remaining missing values after conversion
            if dataframe.isnull().values.any():
                #st.write("Missing values after conversion detected! Filling missing values with column means.")
                dataframe = dataframe.fillna(dataframe.mean())  # Reapply fill for any new NaNs

            # Check for columns with zero variance
            #st.write("Checking for columns with zero variance...")
            zero_variance_columns = dataframe.columns[dataframe.nunique() <= 1]
            if len(zero_variance_columns) > 0:
                #st.write(f"Columns with zero variance: {zero_variance_columns}. Removing these columns.")
                dataframe = dataframe.loc[:, dataframe.nunique() > 1]  # Remove zero-variance columns
            else:
                #st.write("No columns with zero variance found.")

            # Display first few rows of the dataset
                st.subheader("Dataset Preview")
                st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, 0:-1]  # Features (all except the last column)
            Y = array[:, -1]  # Target (last column)

            # Parameters for Repeated Random Test-Train Splits
            n_splits = st.slider("Select number of splits:", 2, 20, 10)
            test_size = st.slider("Select test size proportion:", 0.1, 0.5, 0.33)
            seed = st.number_input("Set random seed:", min_value=0, value=7)

            # Shuffle and split dataset 'n_splits' times
            shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

            # Train the data on a Linear Regression model
            model = LinearRegression()

            # Evaluate using Repeated Random Test-Train Splits
            st.write("Evaluating the model...")
            results = cross_val_score(model, X, Y, cv=shuffle_split)

            # Display results
            st.subheader("Repeated Random Test-Train Split Results")
            mean_accuracy = results.mean()*100
            std_deviation = results.std()*100
            st.write(f"Accuracy: {mean_accuracy:.3f}%")
            st.write(f"Standard Deviation: {std_deviation:.3f}%")

            # Interpretation 
            st.subheader("Interpretation")
        
            # Accuracy interpretation
            if mean_accuracy > 0.8:
                st.write(f"- The model has a **high accuracy** with a value of {mean_accuracy:.3f}. This means that the model explains a significant portion of the variance in the target variable.")
            elif 0.5 <= mean_accuracy <= 0.8:
                st.write(f"- The model has a **moderate accuracy** with a value of {mean_accuracy:.3f}. The model is reasonably good but could be improved with feature engineering or a different model.")
            else:
                st.write(f"- The model has a **low accuracy** with a value of {mean_accuracy:.3f}. The model is not adequately explaining the variance in the target variable. Consider improving the dataset or trying a different approach.")

            # Standard deviation interpretation
            if std_deviation < 0.05:
                st.write(f"- The model is **highly consistent** across splits, as indicated by the low standard deviation of {std_deviation:.3f}. This means the model's performance is stable across different train-test splits.")
            elif 0.05 <= std_deviation <= 0.15:
                st.write(f"- The model is **moderately consistent** with a standard deviation of {std_deviation:.3f}. There is some variation in model performance across different splits, but it is still acceptable.")
            else:
                st.write(f"- The model is **inconsistent** with a high standard deviation of {std_deviation:.3f}. The model's performance varies significantly across different splits. Consider revisiting the model or dataset.")

             # Interpretation 
            st.subheader("Guide")
            
             # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)
           
            #  first column
            with col1:
                st.write("**Accuraccy**")
                st.write("  - 0.8 to 1.0 indicates a strong model performance.")
                st.write("  - 0.5 to 0.8 suggests that the model is performing adequately but may require further refinement.")
                st.write("  - 0.0 to 0.5 indicates poor model performance and potential overfitting or underfitting.")

            # second column
            with col2:
                st.write("**Standard Dveiation**")
                st.write("  - Less than 0.05 suggests very little variability in performance.")
                st.write("  - 0.05 to 0.15 indicates moderate variability in performance, which is generally acceptable for many models.")
                st.write("  - Greater than 0.15 suggests significant variability, which may indicate that the model is not robust across different samples.")
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()


    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['temp', 'hum', 'wind', 'pres', 'heat', 'dew', 'chill', 'temphum', 'humwind', 'target']  # Ensure correct column names
        dataframe = pd.read_csv(uploaded_file, names=names, header=None)
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with missing values (optional step)
        dataframe = dataframe.dropna()
        
        return dataframe

    # Main app
    def main():
        st.title("1. Mean Squared Error (MSE) (80:20 train-test split)")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="unique_file_uploader2")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, :-1]  
            Y = array[:, -1]  

            # Split the dataset into an 80:20 train-test split
            test_size = 0.2
            seed = 42  # Random seed for reproducibility
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the data on a Linear Regression model
            model = LinearRegression()
            model.fit(X_train, Y_train)

            # Make predictions on the test data
            Y_pred = model.predict(X_test)

            # Calculate the mean squared error on the test set
            mse = mean_squared_error(Y_test, Y_pred)

            # Display results
            st.subheader("Model Evaluation")
            st.write(f"Mean Squared Error (MSE): {mse:.3f}")

            # Interpretation and guidance
            st.subheader("Interpretation of Results")

            # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)

            #  first column
            with col1:
                st.write(
                """
                The Mean Squared Error (MSE) is a measure of how well the linear regression model predicts the target variable. 
                Specifically, MSE represents the average squared difference between the predicted values and the actual values. 
                A lower MSE indicates a better fit of the model to the data, meaning that the predictions are closer to the actual outcomes.
                    """
                )
                
            # second column
            with col2:
                st.write(
                    """
                    Here's a general guideline for interpreting the MSE value:
                    - **MSE = 0:** Perfect model with no prediction error.
                    - **0 < MSE < 10:** Good model performance. The predictions are reasonably close to the actual values.
                    - **10 ≤ MSE < 100:** Moderate model performance. The predictions have some error and may require further improvement.
                    - **MSE ≥ 100:** Poor model performance. The predictions are far from the actual values, indicating that the model may not be capturing the underlying trends in the data well.
                    """
                )
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()

    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['temp', 'hum', 'wind', 'pres', 'heat', 'dew', 'chill', 'temphum', 'humwind', 'target']  # Ensure correct column names
        dataframe = pd.read_csv(uploaded_file, names=names, header=None)
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Drop rows with any NaN values
        dataframe = dataframe.dropna()
        
        return dataframe  # Ensure the dataframe is returned

    # Main app
    def main():
        st.title("2. Mean Absolute Error (MAE) (K-fold Cross Validation)")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="unique_file_uploader")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, :-1]  # Features
            Y = array[:, -1]   # Target variable

            # Set up cross-validation
            kfold = KFold(n_splits=10, random_state=None)
            
            # Train the model
            model = LinearRegression()

            # Calculate the mean absolute error
            scoring = 'neg_mean_absolute_error'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            
            # Display results
            st.subheader("Cross-Validation Results")
            mae = -results.mean()
            std_dev = results.std()
            st.write(f"Mean Absolute Error (MAE): {mae:.3f} (+/- {std_dev:.3f})")

            # Interpretation and guidance
            st.subheader("Interpretation of Results")

            # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)

            #  first column
            with col1:
                st.write(
                    """
                    The Mean Absolute Error (MAE) is a measure of prediction accuracy for regression models. 
                    It represents the average absolute difference between the predicted values and the actual values. 
                    A lower MAE indicates better model performance, meaning the predictions are closer to the actual outcomes.
                    """
                )   
                
            # second column
            with col2:
                st.write(
                    """
                    Here's how to interpret the MAE value:
                    - **MAE = 0:** Perfect model with no prediction error.
                    - **0 < MAE < 10:** Good model performance, with predictions reasonably close to the actual values.
                    - **10 ≤ MAE < 50:** Moderate model performance; the predictions have some error and may need improvement.
                    - **MAE ≥ 50:** Poor model performance; the predictions are far from the actual values, indicating that the model may not capture the underlying trends well. """
                )
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()    
    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        names = ['temp', 'hum', 'wind', 'pres', 'heat', 'dew', 'chill', 'temphum', 'humwind', 'target']  # Ensure correct column names
        dataframe = pd.read_csv(uploaded_file, names=names, header=None)
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Drop rows with any NaN values
        dataframe = dataframe.dropna()
        
        return dataframe  # Ensure the dataframe is returned

    # Main app
    def main():
        st.title(" 3. R² (K-fold Cross Validation)")

        # Unique file uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="unique_file_uploader3")

        if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, :-1]  
            Y = array[:, -1]   

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None, shuffle=True)

            # Train the data on a Linear Regression model
            model = LinearRegression()

            # Calculate the R² score using cross-validation
            scoring = 'r2'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

    # Display results
            st.subheader("Model Evaluation")
            st.write(f"Average R² Score: {results.mean():.3f} (± {results.std():.3f})")

            # Interpretation and guidance
            st.subheader("Interpretation of Results")
            st.write("""
            The R² (R-squared) score is a statistical measure that represents the proportion of variance for a dependent variable that's explained by independent variables in a regression model. 

            Here's how to interpret the R² score:
            - **R² = 1:** Perfect model; it explains 100% of the variability in the target variable.
            - **0.7 ≤ R² < 1:** Good model performance; a significant portion of the variance is explained by the model.
            - **0.3 ≤ R² < 0.7:** Moderate model performance; the model explains some variance but may be improved.
            - **R² < 0.3:** Poor model performance; the model explains very little variance, indicating that it may not be suitable for the data.

            Keep in mind that a higher R² value does not always mean the model is better. It's essential to validate the model's performance using other metrics and visualizations.
            """
            )
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()
 #macala normailah itd105 it4d
