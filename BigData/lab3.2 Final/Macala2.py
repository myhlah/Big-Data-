import streamlit as st
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit # sampling techniques
from sklearn.linear_model import LogisticRegression, LinearRegression  # ML Algorithm 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score #performance metric
from sklearn.tree import DecisionTreeRegressor #Hyperparameter Tuning
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR



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
option = st.sidebar.radio("Part II - Regression Task", ['Sampling Technique/s', 'Performance Metric/s','Hyperparameter Tuning'])
st.sidebar.write("""  
    ### Instructions:
    1. Upload your CSV dataset.
    2. Click the button to evaluate the model.
      """)

# Regression Task
if option == 'Sampling Technique/s':
    st.title("Water Quality Prediction Model")

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
    st.write("<br><br>", unsafe_allow_html=True)

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

    def main():
        #Split into Train and Test Sets
        st.title("Split into Train and Test Sets")
        uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv", key="file_uploader_unique_key")

        if uploaded_file is not None:
            # Read dataset
            dataframe = load_water_data(uploaded_file)

            # Normalize column names
            dataframe.columns = dataframe.columns.str.strip().str.lower()  # Lowercase and strip whitespace

            # Debugging: Check the shape and columns of the dataset
            st.write("### Shape of the Dataset:")
            st.write(dataframe.shape)  # Show number of rows and columns

            st.write("### Dataset Preview:")
            st.write(dataframe.head())  # Show the first few rows of the dataset

            #st.write("### Dataset Columns:")
            #st.write(dataframe.columns.tolist())  # Show column names as a list

            #st.write("### Check for Missing Values:")
            #st.write(dataframe.isnull().sum())
            st.write("### Correlation Matrix:")
            st.write(dataframe.corr())

            # Check for any non-numeric columns
            non_numeric_cols = dataframe.select_dtypes(include=['object']).columns
            if len(non_numeric_cols) > 0:
                st.write(f"Non-numeric columns found: {non_numeric_cols}. Please ensure only numeric data is used for model training.")

            array = dataframe.values
            X = array[:, 0:20]  
            Y = array[:, 20]   

            st.write("### Shape of the Dataset:")
            st.write(dataframe.shape)  # This will show you how many rows and columns are in your dataset

            # Split into input and output variables
            X = dataframe.drop('is_safe', axis=1).values
            Y = dataframe['is_safe'].values

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

            # Interpretation of the accuracy
            if accuracy < 0:
                st.warning("The model's performance is poor. Please check your data for quality and relevance.")
            elif accuracy < 50:
                st.warning("The model has low predictive power. Consider gathering more data or using different features.")
            elif accuracy < 75:
                st.success("The model has moderate predictive power. It may perform reasonably well.")
            else:
                st.success("The model has high predictive power.")

            # Save and download the model
            model.fit(X, Y)
            model_filename = "logistic_regression_split_regression.joblib"
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
  
####################################################################################################
####################################################################################################

# Performance Metric
elif option == 'Performance Metric/s':
    st.title("PERFORMANCE METRICS")
    # Introduction and Instructions
    st.write("""       
        ### Instructions:
        1. Upload your CSV dataset.
        2. Click the button to evaluate the model.
        """)
    st.write("<br><br>", unsafe_allow_html=True)

    @st.cache_data
    def load_data(uploaded_file):
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
    

    def main():
        st.title("Mean Squared Error (MSE) (80:20 train-test split)")

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
        st.write("<br><br>", unsafe_allow_html=True)

    if __name__ == "__main__":
        main()
    
    def main():
        st.title("Mean Absolute Error (MAE) (K-fold Cross Validation)")

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
                - **MAE ≥ 50:** Poor model performance; the predictions are far from the actual values, indicating that the model may not capture the underlying trends well. 
                """
            )
        else:
            st.write("Please upload a CSV file to proceed.")
        st.write("<br><br>", unsafe_allow_html=True)    

    if __name__ == "__main__":
        main()    

####################################################################################################
####################################################################################################

# Hyperparameter Tuning
elif option == 'Hyperparameter Tuning':
    st.title("HYPERPARAMETER TUNING")
    st.write("""       
        ### Instructions for Hyperparameter Tuning
        1. **Set Hyperparameters**  
        
        2. **Run Tuning**  
        
        3. **Results**  
    """)

    st.write("<br><br>", unsafe_allow_html=True)
    # Initialize accuracy dictionary
    maeresults = {}

    def main():
        # Load the dataset
        filename = 'E:\Myla docs\programs react\ITD105\lab3\water.csv'
        dataframe = pd.read_csv(filename)

        # Ensure all columns are numeric, and handle non-numeric entries by replacing them with NaN
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        dataframe = dataframe.dropna()

        # Separate features and target
        X = dataframe.iloc[:, :-1].values
        Y = dataframe.iloc[:, -1].values
            
        # Tabs 
        tabs = st.tabs(["CART", "Elastic Net", "AdaBoost","K-NN","Lasso","Ridge","Linear","MLP","Random Forest","SVM" ])

        with tabs[0]:
            # Hyperparameter input fields in the main content area
            st.header("CART (Classification and Regression Trees) - Decision Tree Regressor")
            st.subheader("Set Hyperparameters")
            max_depth = st.slider("Max Depth", 1, 20, None, key="cart_max_depth")
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2, key="cart_min_samples_split")
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1, key="cart_min_samples_leaf")
            n_splits = st.slider("Number of Folds (K)", 2, 20, 10, key="cart_n_splits")


            # Split the dataset into K-Folds
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Create and train the Decision Tree Regressor
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

            # Calculate the mean absolute error using cross-validation
            scoring = 'neg_mean_absolute_error'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

            # Display the results in the app
            mae = -results.mean()
            mae_std = results.std()
            maeresults["CART"] = mae

            st.write(f"Mean Absolute Error (MAE): {mae:.3f} ± {mae_std:.3f}")
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Reflects the average magnitude of errors in predictions. Lower MAE indicates better model performance.
               
                - **Max Depth**: Increasing this value allows the model to capture more details in the data but may risk overfitting.
                
                - **Min Samples Split and Leaf**: Adjust these to balance complexity and overfitting; higher values simplify the model.
                
            """)

        with tabs[1]:
            # Hyperparameter input fields in the main content area
            st.header("Elastic Net")
            st.subheader("Set Hyperparameters")
            alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1, key="elastic_alpha")
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01, key="elastic_l1_ratio")
            max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100, key="elastic_max_iter")


            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the Elastic Net model
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=None)

            # Calculate the mean absolute error
            scoring = 'neg_mean_absolute_error'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            maeresults["Elastic Net"] = -results.mean()
            # Display the results
            st.write(f"Mean Absolute Error (MAE): {-results.mean():.3f} ± {results.std():.3f}")
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Measures prediction accuracy. Lower values are better.
                
                - **Alpha and L1 Ratio**: Control regularization strength and balance. Adjust to manage overfitting and underfitting.
                     
                """)


        with tabs[2]:
            # Hyperparameter input fields in the main content area
            st.header("Gradient Boosting Machines (AdaBoost) Regressor")
            st.subheader("Set Hyperparameters")
            n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1)
            learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01)

            # K-Fold Cross-Validation
            kfold = KFold(n_splits=10, random_state=42, shuffle=True)

            # Initialize AdaBoost Model
            ada_model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

            # Cross-Validation for Mean Absolute Error
            scoring = 'neg_mean_absolute_error'
            ada_results = cross_val_score(ada_model, X, Y, cv=kfold, scoring=scoring)

            # Calculate and Display Results
            mean_mae = -ada_results.mean()
            std_dev_mae = ada_results.std()
            maeresults["AdaBoost"] = mean_mae
            st.write("<br>", unsafe_allow_html=True)

            st.write(f"Mean Absolute Error (MAE)**: {mean_mae:.3f} ± {std_dev_mae:.3f}")
            st.write("<br>", unsafe_allow_html=True)
            # Interpretation and Guide for Results
            st.write("""
                     
               - **Mean Absolute Error (MAE)**: Indicates the average prediction error. Lower MAE is preferred.
            
               - **Number of Estimators**: More estimators may improve accuracy but increase computational cost.
                
               - **Learning Rate**: Lower values stabilize learning; adjust for performance consistency.
   
            """)

        with tabs[3]:
            # Hyperparameter input fields in the main content area
            st.header("K-Nearest Neighbors (K-NN) ")
            st.subheader("Set Hyperparameters")

            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
            weights = st.selectbox("Weights", ["uniform", "distance"])
            algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the K-NN model
            knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

            # Calculate the mean absolute error with K-NN
            scoring = 'neg_mean_absolute_error'
            knn_results = cross_val_score(knn_model, X, Y, cv=kfold, scoring=scoring)
            maeresults["K-NN"] = -knn_results.mean() 
            st.write("<br>", unsafe_allow_html=True)
            # Display the results
            st.write(f"K-NN Mean Absolute Error (MAE): {-knn_results.mean():.3f} ± {knn_results.std():.3f}")
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Reflects prediction accuracy. Aim for lower values.
                     
                - **Number of Neighbors**: Smaller values capture local patterns; larger values smooth predictions.
                     
                - **Weights and Algorithm**: Adjust for data characteristics and computational efficiency.
                     
                """)

        with tabs[4]:
            # Hyperparameter input fields in the main content area
            st.header("Lasso Regression ")
            st.subheader("Set Hyperparameters")

            alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key="lasso_alpha")
            max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key="lasso_max_iter")

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the data on a Lasso Regression model
            lasso_model = Lasso(alpha=alpha, max_iter=max_iter, random_state=None)

            # Calculate the mean absolute error with Lasso
            scoring = 'neg_mean_absolute_error'
            lasso_results = cross_val_score(lasso_model, X, Y, cv=kfold, scoring=scoring)
            st.write("<br>", unsafe_allow_html=True)
            # Display Lasso results
            maeresults["Lasso"] = -lasso_results.mean()
            st.write(f"Lasso Mean Absolute Error (MAE): {-lasso_results.mean():.3f} ± {lasso_results.std():.3f}")
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Shows average prediction error. Lower is better.
                     
                - **Alpha**: Higher values increase regularization, simplifying the model but possibly underfitting.
                     
                """)

        with tabs[5]:
            # Hyperparameter input fields in the main content area
            st.header("Ridge Regression ")
            st.subheader("Set Hyperparameters")

            alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key="ridge_alpha")
            max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key="ridge_max_iter")

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the data on a Ridge Regression model
            ridge_model = Ridge(alpha=alpha, max_iter=max_iter, random_state=None)

            # Calculate the mean absolute error with Ridge
            ridge_results = cross_val_score(ridge_model, X, Y, cv=kfold, scoring=scoring)
            st.write("<br>", unsafe_allow_html=True)
            # Display Ridge results
            maeresults["Ridge"] = -ridge_results.mean()
            st.write(f"Ridge Mean Absolute Error (MAE): {-ridge_results.mean():.3f} ± {ridge_results.std():.3f}")
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Indicates prediction accuracy. Lower values are desirable.
                     
                - **Alpha**: Controls regularization. Adjust to balance model simplicity and predictive power.
                     
                """)


        with tabs[6]:
            # Hyperparameter input fields in the main content area
            st.header("Linear Regression")
            st.subheader("Set Hyperparameters")

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the data on a Linear Regression model
            model = LinearRegression()

            # Calculate the mean absolute error
            scoring = 'neg_mean_absolute_error'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            maeresults["Linear"] = -results.mean()
            st.write("<br>", unsafe_allow_html=True)
            # Display results
            st.write(f"Mean Absolute Error (MAE): {-results.mean():.3f} ± {results.std():.3f}")
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Reflects prediction errors. A lower MAE signifies a better fit.
                     
                - **Performance Insights**: Compare MAE to the target range for evaluating accuracy.
                     
                """)

        with tabs[7]:
            # Hyperparameter input fields in the main content area
            st.header("Multi-Layer Perceptron (MLP) Regressor")
            st.subheader("Set Hyperparameters")

            hidden_layer_sizes = st.slider("Hidden Layer Sizes", min_value=10, max_value=200, value=(100, 50), step=10)
            activation = st.selectbox("Activation Function", options=['identity', 'logistic', 'tanh', 'relu'], index=3)
            solver = st.selectbox("Solver", options=['adam', 'lbfgs', 'sgd'], index=0)
            learning_rate = st.selectbox("Learning Rate Schedule", options=['constant', 'invscaling', 'adaptive'], index=0)
            max_iter = st.slider("Max Iterations", min_value=100, max_value=2000, value=1000, step=100, key="max1")
            random_state = st.number_input("Random State", value=50)

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the data on an MLP Regressor with specified hyperparameters
            mlp_model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,  # Use user-defined hidden layer sizes
                activation=activation,                   # Use user-defined activation function
                solver=solver,                          # Use user-defined optimization algorithm
                learning_rate=learning_rate,            # Use user-defined learning rate schedule
                max_iter=max_iter,                      # Use user-defined max iterations
                random_state=random_state                # Use user-defined random state
            )

            # Calculate the mean absolute error with MLP
            scoring = 'neg_mean_absolute_error'
            mlp_results = cross_val_score(mlp_model, X, Y, cv=kfold, scoring=scoring)
            maeresults["MLP"] = -mlp_results.mean()
            st.write("<br>", unsafe_allow_html=True)
            # Display results
            st.write("Mean Absolute Error (MAE): %.3f ± %.3f" % (-mlp_results.mean(), mlp_results.std()))
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Reflects model prediction accuracy. Lower is preferable.
                     
                - **Hidden Layers and Activation**: Adjust for complexity and pattern recognition.
                     
                - **Solver and Learning Rate**: Balance learning speed and stability.
                     
                """)

        with tabs[8]:
            # Hyperparameter input fields in the main content area
            st.header("Random Forest Regressor")
            st.subheader("Set Hyperparameters")

            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None)
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
            min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1)
            random_state = st.number_input("Random State", value=42)

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the data on a Random Forest Regressor with specified hyperparameters
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators,      # Use user-defined number of trees
                max_depth=max_depth,            # Use user-defined max depth
                min_samples_split=min_samples_split,  # Use user-defined min samples split
                min_samples_leaf=min_samples_leaf,    # Use user-defined min samples leaf
                random_state=random_state        # Use user-defined random state
            )

            # Calculate the mean absolute error with Random Forest
            scoring = 'neg_mean_absolute_error'
            rf_results = cross_val_score(rf_model, X, Y, cv=kfold, scoring=scoring)
            maeresults["Random Forest"] = -rf_results.mean()
            st.write("<br>", unsafe_allow_html=True)
            # Display results
            st.write("Random Forest Mean Absolute Error (MAE): %.3f ± %.3f" % (-rf_results.mean(), rf_results.std()))
            st.write("<br>", unsafe_allow_html=True)
            st.write("""
                     
                - **Mean Absolute Error (MAE)**: Measures average error in predictions. Lower MAE is ideal.
                     
                - **Number of Trees and Depth**: Control model complexity and prediction accuracy. Higher values may improve predictions but increase computation time.
                     
                - **Samples Split and Leaf**: Adjust to manage overfitting and performance.
                     
                """)

        with tabs[9]:
            # Hyperparameter input fields in the main content area
            st.header("Support Vector Machines (SVM) Regressor ")
            st.subheader("Set Hyperparameters")
            
            kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
            C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
            epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            # Split the dataset into a 10-fold cross-validation
            kfold = KFold(n_splits=10, random_state=None)

            # Train the data on a Support Vector Regressor with specified hyperparameters
            svm_model = SVR(
                kernel=kernel,          # Use user-defined kernel
                C=C,                    # Use user-defined regularization parameter
                epsilon=epsilon         # Use user-defined epsilon
            )

            # Calculate the mean absolute error with SVM
            scoring = 'neg_mean_absolute_error'
            svm_results = cross_val_score(svm_model, X, Y, cv=kfold, scoring=scoring)
            maeresults["SVM"] = -svm_results.mean()
            st.write("<br>", unsafe_allow_html=True)
            # Display results
            st.write("SVM Mean Absolute Error (MAE): %.3f ± %.3f" % (-svm_results.mean(), svm_results.std()))
            st.write("<br>", unsafe_allow_html=True)
            st.write("""

                - **Mean Absolute Error (MAE)**: Represents the average magnitude of errors in the model's predictions. A smaller MAE indicates higher prediction accuracy.
                     
                - **Standard Deviation of MAE**: Reflects the consistency of the model's performance across different validation folds. A smaller standard deviation implies more consistent performance.
                     
                - **Kernel Choice**: 
                    - Use **linear** for datasets that are linearly separable.
                    - Use **rbf** (Radial Basis Function) for non-linear datasets as it is versatile and effective for most applications.
                     
                - **Regularization Parameter (C)**: 
                    - Higher values of `C` allow the model to prioritize minimizing prediction errors but may lead to overfitting.
                    - Lower values promote a simpler model at the risk of underfitting.
                     
                - **Epsilon**: 
                    - Controls the margin of tolerance where predictions are considered acceptable. Smaller values make the model more sensitive to prediction errors.
                
                """)
        st.write("<br><br>", unsafe_allow_html=True)
        # MAE Results Table and Visualization
        st.subheader("Model Comparison")
        mae_df = pd.DataFrame(list(maeresults.items()), columns=["Algorithm", "MAE"])
        mae_df["MAE"] = pd.to_numeric(mae_df["MAE"], errors='coerce')
        mae_df = mae_df.dropna(subset=["MAE"])
        highest = mae_df["MAE"].idxmax()
        lowest = mae_df["MAE"].idxmin()
        
        # Highlight highest and lowest
        def highlight_row(row):
            if row.name == highest:
                return ["background-color: red; color: white"] * len(row)
            elif row.name == lowest:
                return ["background-color: blue; color: white"] * len(row)
            else:
                return [""] * len(row)

        st.dataframe(mae_df.style.apply(highlight_row, axis=1))
        st.write("<br><br>", unsafe_allow_html=True)
        # Visualization
        colors = ['blue' if idx == lowest else 'red' if idx == highest else 'green' for idx in range(len(mae_df))]

        #st.bar_chart(data=mae_df.set_index("Algorithm"), use_container_width=True)
        fig, ax = plt.subplots()
        ax.bar(mae_df["Algorithm"], mae_df["MAE"], color=colors)
        ax.set_title("MAE Comparison of ML Algorithms")
        ax.set_ylabel("MAE (%)")
        ax.set_xlabel("Algorithm")
        plt.xticks(rotation=45, ha="right")

        st.pyplot(fig)

        # Insights
        st.write("### Insights")
        st.write("""
        - **The highlighted blue algorithm had the lowest MAE, which is generally better.
        - **The highlighted red algorithm had the highest MAE.
         """)

    # Call main function to run code in Streamlit
    if __name__ == "__main__":
        main()
 #macala normailah itd105 it4d lab 3 part 2
