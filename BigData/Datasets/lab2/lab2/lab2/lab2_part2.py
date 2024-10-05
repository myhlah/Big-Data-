import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # ML Algorithm used
import joblib  # Import joblib to save and load the model
import numpy as np
import pandas as pd

st.title("Climate Prediction Model")

# Introduction and Instructions
st.write("""
Part II - Regression Task using Train-Test Split       
### Instructions:
1. Upload your CSV dataset.
2. Click the button to evaluate the model.
""")

# RESAMPLING TECHNIQUES
st.title("RESAMPLING TECHNIQUES")
# Split into Train and Test Sets
st.title("Split into Train and Test Sets")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv", key="file_uploader_unique_key")

if uploaded_file is not None:
    # Read dataset
    data = pd.read_csv(uploaded_file)

    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()  # Lowercase and strip whitespace

    # Debugging: Check the shape and columns of the dataset
    st.write("### Shape of the Dataset:")
    st.write(data.shape)  # Show number of rows and columns

    st.write("### Dataset Preview:")
    st.write(data.head())  # Show the first few rows of the dataset

    st.write("### Dataset Columns:")
    st.write(data.columns.tolist())  # Show column names as a list

    st.write("### Check for Missing Values:")
    st.write(data.isnull().sum())
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
    pres = st.number_input("Pressure (1-100%)", min_value=1, max_value=100, value=0)  
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
        st.subheader("Interpretation of the Prediction")

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


           
####################################################################################################
####################################################################################################
### 

### 2. Repeated Random Test-Train Splits
import streamlit as st
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

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

        # Check for missing values
        #st.write("Checking for missing values...")
         #st.write(dataframe.isnull().sum())
        if dataframe.isnull().values.any():
            st.write("Missing values detected! Filling missing values with column means.")
             #dataframe = dataframe.fillna(dataframe.mean())  # Fills missing values with column mean

        # Check data types
         #st.write("Checking data types...")
         #st.write(dataframe.dtypes)

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

        # Interpretation and Guide
        st.subheader("Interpretation and Guide")

        # Accuracy interpretation
        if mean_accuracy > 0.8:
            st.write(f"The model has a **high accuracy** with a value of {mean_accuracy:.3f}. This means that the model explains a significant portion of the variance in the target variable.")
        elif 0.5 <= mean_accuracy <= 0.8:
            st.write(f"The model has a **moderate accuracy** with a value of {mean_accuracy:.3f}. The model is reasonably good but could be improved with feature engineering or a different model.")
        else:
            st.write(f"The model has a **low accuracy** with a value of {mean_accuracy:.3f}. The model is not adequately explaining the variance in the target variable. Consider improving the dataset or trying a different approach.")

        # Standard deviation interpretation
        if std_deviation < 0.05:
            st.write(f"The model is **highly consistent** across splits, as indicated by the low standard deviation of {std_deviation:.3f}. This means the model's performance is stable across different train-test splits.")
        elif 0.05 <= std_deviation <= 0.15:
            st.write(f"The model is **moderately consistent** with a standard deviation of {std_deviation:.3f}. There is some variation in model performance across different splits, but it is still acceptable.")
        else:
            st.write(f"The model is **inconsistent** with a high standard deviation of {std_deviation:.3f}. The model's performance varies significantly across different splits. Consider revisiting the model or dataset.")

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()


####################################################################################################
####################################################################################################
### 

### Performance Metrics Here
st.title("PERFORMANCE METRICS")
### 1. Mean Squared Error (MSE) (80:20 train-test split)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
        X = array[:, :-1]  # Features (all columns except the last one)
        Y = array[:, -1]   # Target variable (last column)

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
        st.write(
            """
            The Mean Squared Error (MSE) is a measure of how well the linear regression model predicts the target variable. 
            Specifically, MSE represents the average squared difference between the predicted values and the actual values. 
            A lower MSE indicates a better fit of the model to the data, meaning that the predictions are closer to the actual outcomes.

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


####################################################################################################
####################################################################################################
### 

### 2. Mean Absolute Error (MAE) (K-fold Cross Validation)

import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

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
        st.write(
            """
            The Mean Absolute Error (MAE) is a measure of prediction accuracy for regression models. 
            It represents the average absolute difference between the predicted values and the actual values. 
            A lower MAE indicates better model performance, meaning the predictions are closer to the actual outcomes.

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


####################################################################################################
####################################################################################################
### 

### 3. R² (K-fold Cross Validation)

import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

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
    st.title("R² (K-fold Cross Validation)")

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
        X = array[:, :-1]  # Features (all columns except the last one)
        Y = array[:, -1]   # Target variable (last column)

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
        st.write(
            """
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
