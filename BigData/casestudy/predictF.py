import streamlit as st
import pandas as pd
import joblib

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = 0  # Track the current question index
if 'answers' not in st.session_state:
    st.session_state['answers'] = {}

# Define all the questions in order with their respective options
questions = [
    ("Enter your name (Optional):", "text_input"),
    ("Do you have high blood pressure?", "radio", [0, 1]),
    ("Do you have high cholesterol?", "radio", [0, 1]),
    ("Have you had a cholesterol check in the last 5 years?", "radio", [0, 1]),
    ("What is your Body Mass Index (BMI)?", "number_input", 0, 120, 0),
    ("Have you ever smoked at least 100 cigarettes in your entire life?", "radio", [0, 1]),
    ("Have you ever been told by a doctor that you had a stroke?", "radio", [0, 1]),
    ("Have you ever been diagnosed with coronary heart disease (CHD) or had a myocardial infarction (MI)?", "radio", [0, 1]),
    ("Have you engaged in physical activity in the past 30 days, excluding your job?", "radio", [0, 1]),
    ("Do you consume fruits at least once per day?", "radio", [0, 1]),
    ("Do you consume vegetables at least once per day?", "radio", [0, 1]),
    ("Do you consume alcohol heavily?", "radio", [0, 1]),
    ("Do you have any kind of health care coverage?", "radio", [0, 1]),
    ("In the past 12 months, was there ever a time when you needed to see a doctor but could not because of cost?", "radio", [0, 1]),
    ("How would you rate your general health?", "radio", [1, 2, 3, 4, 5]),  # This will remain with 5 options
    ("In the past 30 days, how many days did you experience poor mental health? (0-30)", "number_input", 0, 30, 0),
    ("Do you have any physical health issues?", "radio", [0, 1]),
    ("Do you have serious difficulty walking or climbing stairs?", "radio", [0, 1]),
    ("What is your gender?", "radio", [0, 1]),
    ("What is your age category?", "radio", [1, 2, 3, 4]),
    ("What is your highest level of education?", "radio", [1, 2, 3, 4, 5, 6]),
    ("What is your annual income?", "radio", [1, 2, 3, 4, 5, 6])
]

# Function to display the current question
def display_question(index):
    if index >= len(questions):  # Ensure index is within bounds
        return

    question, qtype, *args = questions[index]
    
    if qtype == "text_input":
        user_input = st.text_input(question)
    elif qtype == "radio":
        # Radio options are based on the provided choices
        if question == "How would you rate your general health?":
            options = ["Excellent", "Very good", "Good", "Fair", "Poor"]
        elif question == "What is your gender?":
            options = ["Female", "Male"]    
        elif question == "What is your age category?":
            options = ["18-30", "31-59", "60-79", "80 or older"]
        elif question == "What is your highest level of education?":
            options = ["No formal education/Kindergarten Only", "Elementary", "High School", "Senior High", "College", "Masters/Doctorate"]
        elif question == "What is your annual income?":
            options = ["Below 100,000", "100,000-300,000", "300,000-600,000", "600,000-1,000,000", "1,000,000-3,000,000", "Above 3,000,000"]
        else:
            options = ["Yes", "No"]
        
        user_input = st.radio(question, options, index=0)
    elif qtype == "number_input":
        min_value, max_value, value = args
        user_input = st.number_input(question, min_value=min_value, max_value=max_value, value=value)
    
    # Store the answer in session state
    st.session_state['answers'][f"q{index}"] = user_input

# Function to display the input summary after all questions are answered
def display_summary():
    st.write("### Input Summary:")
    # Display answers in a table format
    data = []
    for i, (question, qtype, *args) in enumerate(questions):
        answer = st.session_state['answers'].get(f"q{i}")
        data.append([question, answer])
    
    # Create a table with the collected answers
    df = pd.DataFrame(data, columns=["Question", "Answer"])
    st.dataframe(df)

# Function to reset the app to start over
def reset_session():
    st.session_state['current_question'] = 0
    st.session_state['answers'] = {}

def convert_answers_to_numeric():
    numeric_answers = []
    for i, (question, qtype, *args) in enumerate(questions):
        answer = st.session_state['answers'].get(f"q{i}")
        
        if isinstance(answer, str):  # If the answer is a string, check which question it is
            if question == "How would you rate your general health?":
                options = ["Excellent", "Very good", "Good", "Fair", "Poor"]
                # Ensure the answer is in the options list
                numeric_answers.append(options.index(answer))  # Convert the answer to an index
            elif question == "What is your gender?":
                options = ["Female", "Male"]
                numeric_answers.append(options.index(answer))
            elif question == "What is your age category?":
                options = ["18-30", "31-59", "60-79", "80 or older"]
                numeric_answers.append(options.index(answer))
            elif question == "What is your highest level of education?":
                options = ["No formal education/Kindergarten Only", "Elementary", "High School", "Senior High", "College", "Masters/Doctorate"]
                numeric_answers.append(options.index(answer))
            elif question == "What is your annual income?":
                options = ["Below 100,000", "100,000-300,000", "300,000-600,000", "600,000-1,000,000", "1,000,000-3,000,000", "Above 3,000,000"]
                numeric_answers.append(options.index(answer))
            else:
                # Convert 'Yes' to 1 and 'No' to 0
                if answer == "Yes":
                    numeric_answers.append(1)
                elif answer == "No":
                    numeric_answers.append(0)
        else:
            numeric_answers.append(answer)  # For numeric input types like BMI, health days, etc.
    
    return numeric_answers

def main():
    # Streamlit App Title
    st.title("GlucoSense")
    st.subheader("Make Predictions using a Trained Model")
    st.write("GlucoSense is an innovative tool designed to assess your health and provide insights based on your responses. "
            "It uses a trained machine learning model to make predictions about potential health conditions.")
    st.write("**Instruction:** Answer the following questions to the best of your ability. Click 'Submit Answer' to move to the next question. Once you've completed all questions, click 'Predict' to view the result.")

    
     # Load the embedded model
    model_path = "trained_model.pkl"  # Replace with the relative path to your model file
    try:
        model = joblib.load(model_path)
        st.success("Model successfully loaded!")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the correct directory.")
        return

    # Display the current question based on session state
    display_question(st.session_state['current_question'])

    # Calculate the progress (e.g., 1 out of 22 questions)
    progress = min((st.session_state['current_question'] + 1) / len(questions), 1.0)  # Ensure progress doesn't exceed 1.0
    st.progress(progress)
    
    # Handle automatic progression through the questions
    if st.button("Submit Answer"):
        # Proceed to the next question
        st.session_state['current_question'] += 1
        
        # If all questions are answered, show the input summary and prediction button
        if st.session_state['current_question'] == len(questions):
            display_summary()

            # Convert answers to numeric format
            input_data = convert_answers_to_numeric()

            # Make a prediction using the model
            prediction = model.predict([input_data])
            if prediction == 1:
                st.success("Prediction: Likely to have Heart Disease")
            else:
                st.success("Prediction: Unlikely to have Heart Disease")
        else:
            st.rerun()  # Re-run the app to display the next question

    # Reset button to restart the input process
    if st.button("Reset"):
        reset_session()
        st.rerun()

if __name__ == "__main__":
    main()

#normailah macala 04/12/2024
