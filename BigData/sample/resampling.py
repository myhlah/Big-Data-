### 1. Split into Train and Test Sets
import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # Import joblib to save and load the model
import numpy as np

# Set title for the app
st.title("Diabetes Prediction Model")

st.title("1. Split into Train and Test Sets")

# File uploader for dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(uploaded_file, names=names)
    st.write("Dataset:")
    st.write(dataframe.head())

    # Split into input and output variables
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]

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
    st.write(f"Accuracy: {result * 100:.3f}%")


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
        st.subheader("Input Sample Data for Prediction")
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        plas = st.number_input("Plasma Glucose Concentration", min_value=0, max_value=200, value=0)
        pres = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
        test = st.number_input("Insulin Level", min_value=0, max_value=800, value=0)
        mass = st.number_input("BMI", min_value=0.0, max_value=60.0, value=0.0)
        pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
        age = st.number_input("Age", min_value=0, max_value=120, value=0)

        input_data = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])

        if st.button("Predict"):
            prediction = model.predict(input_data)
            st.subheader("Prediction Result")

            # Check the prediction and display the result
            if prediction[0] == 0:
                st.write("The predicted class is: 0 (Without Diabetes)")
            else:
                st.write("The predicted class is: 1 (With Diabetes)")


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
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("2. K-fold Cross Validation")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load dataset
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)
        
        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

        # Set k for the number of folds
        num_folds = st.slider("Select number of folds for KFold Cross Validation:", 2, 10, 5)

        # Split the dataset into k folds
        kfold = KFold(n_splits=num_folds, shuffle=False, random_state=None)

        # Train the data on a Logistic Regression model
        model = LogisticRegression(max_iter=210)

        # Evaluate using cross-validation
        st.write("Evaluating the model...")
        results = cross_val_score(model, X, Y, cv=kfold)

        # Display results
        st.subheader("Cross-Validation Results")
        st.write(f"Accuracy: {results.mean()*100:.3f}%")
        st.write(f"Standard Deviation: {results.std()*100:.3f}%")
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################

### 3. Leave-One-Out Cross Validation (LOOCV)

import streamlit as st
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("3. Leave-One-Out Cross Validation (LOOCV)")

    # File uploader for CSV with a unique key
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader_unique_key")

    if uploaded_file is not None:
        # Load dataset
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

        # Split dataset into Leave One Out Cross Validation
        loocv = LeaveOneOut()

        # Train the data on a Logistic Regression model
        model = LogisticRegression(max_iter=500)

        # Evaluate using LOOCV
        st.write("Evaluating the model...")
        results = cross_val_score(model, X, Y, cv=loocv)

        # Display results
        st.subheader("Leave One Out Cross-Validation Results")
        st.write(f"Accuracy: {results.mean()*100:.3f}%")
        st.write(f"Standard Deviation: {results.std()*100:.3f}%")
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################

### 4. Repeated Random Test-Train Splits

import streamlit as st
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("4. Repeated Random Test-Train Splits")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader_shuffle")

    if uploaded_file is not None:
        # Load dataset
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

        # Parameters for Repeated Random Test-Train Splits
        n_splits = st.slider("Select number of splits:", 2, 20, 10)
        test_size = st.slider("Select test size proportion:", 0.1, 0.5, 0.33)
        seed = st.number_input("Set random seed:", min_value=0, value=7)

        # Shuffle and split dataset 'n_splits' times
        shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

        # Train the data on a Logistic Regression model
        model = LogisticRegression(max_iter=300)

        # Evaluate using Repeated Random Test-Train Splits
        st.write("Evaluating the model...")
        results = cross_val_score(model, X, Y, cv=shuffle_split)

        # Display results
        st.subheader("Repeated Random Test-Train Split Results")
        st.write(f"Accuracy: {results.mean()*100:.3f}%")
        st.write(f"Standard Deviation: {results.std()*100:.3f}%")
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################

### 5. Time Series Split (Rolling Cross-Validation)

import streamlit as st
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file, parse_dates=['Date'])
    return dataframe

# Main app
def main():
    st.title("5. Time Series Split (Rolling Cross-Validation) on Stock Data")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file (Date, Close)", type=["csv"], key="file_uploader")
    
    if uploaded_file is not None:
        # Load dataset
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Ensure the CSV has the required columns
        if 'Date' in dataframe.columns and 'Close' in dataframe.columns:
            # Preparing the data
            dataframe = dataframe.sort_values('Date')  # Sort by date just in case
            X = dataframe.index.values.reshape(-1, 1)  # Using index as a feature for demonstration
            y = dataframe['Close'].values

            # Parameters for TimeSeriesSplit
            n_splits = st.slider("Select number of splits:", 2, 5, 3)

            # Initialize the model
            model = LinearRegression()

            # TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)

            st.write("Performing Rolling Cross-Validation...")

            # Perform rolling cross-validation
            results = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train the model on the training set
                model.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = model.predict(X_test)

                # Evaluate the model using Mean Squared Error
                mse = mean_squared_error(y_test, y_pred)
                results.append(mse)

                # Display each split's results
                st.write(f"Train on indices: {train_index}, Test on indices: {test_index}")
                st.write(f"Mean Squared Error: {mse:.4f}")

            # Summary of the results
            st.subheader("Cross-Validation Summary")
            st.write(f"Mean MSE across all splits: {sum(results) / len(results):.4f}")
        else:
            st.error("CSV file must contain 'Date' and 'Close' columns.")
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
