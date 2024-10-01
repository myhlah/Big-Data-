### 1.1 Classification Accuracy (K-fold Cross Validation)

import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.title("CLASSIFICATION METRICS")

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("1.1 Classification Accuracy (K-fold Cross Validation)")

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
        num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 20, 10)

        # Split the dataset into k folds
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=None)

        # Train the data on a Logistic Regression model
        model = LogisticRegression(max_iter=210)

        # Calculate the classification accuracy
        scoring = 'accuracy'
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

        # Display results
        st.subheader("Cross-Validation Results")
        st.write(f"Mean Accuracy: {results.mean() * 100:.3f}%")
        st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.boxplot(results)
        plt.title('K-Fold Cross-Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks([1], [f'{num_folds}-Fold'])
        st.pyplot(plt)

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################
### 1.2 Classification Accuracy (split train-test 75:25 split ratio)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
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

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

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

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################
### 2. Logarithmic Loss (Log Loss)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
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

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

        # Split the dataset into a 10-fold cross validation
        num_folds = 10
        kfold = KFold(n_splits=num_folds, random_state=None)

        # Train the data on a Logistic Regression model
        model = LogisticRegression(max_iter=200)

        # Calculate the log-loss for each fold
        scoring = 'neg_log_loss'
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

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
        plt.ylim(0, log_loss_values.max() + 0.5)  # Adjusting y-axis for better visualization
        st.pyplot(plt)  # Display the plot in Streamlit

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################
### 3. Confusion Matrix

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
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

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

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
        st.write("Confusion Matrix:")
        st.write(matrix)

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)  # Display the plot in Streamlit

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()



####################################################################################################
####################################################################################################
### 4. Classification Report

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
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

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

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
        st.text(classification_report(Y_test, predicted))  # Display as text

        # Optionally, display the report as a DataFrame
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()


####################################################################################################
####################################################################################################
### 5. Area Under ROC Curve (ROC AUC)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
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

        # Display first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(dataframe.head())

        # Preparing the data
        array = dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

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

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################
### 1. Mean Absolute Error (MAE) (K-fold Cross Validation)

st.title("REGRESSION METRICS")

import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("1. Mean Absolute Error (MAE) (K-fold Cross Validation)")

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
        st.write(f"MAE: {-results.mean():.3f} (+/- {results.std():.3f})")
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################
### 2. Mean Squared Error (MSE)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

# Main app
def main():
    st.title("2. Mean Squared Error (MSE) (80:20 train-test split)")

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
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

####################################################################################################
####################################################################################################
### 3. R² (K-fold Cross Validation)

import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

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
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
