import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# 1. Loading the dataset from a public URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/DeekshithaRudrapati/Sleep_Disorder_Prediction/main/Sleep_health_and_lifestyle_dataset.csv"  # Replace with your dataset URL
    data = pd.read_csv(url)
    return data

# 2. Preprocess data (encoding, handling missing values, and splitting)
def preprocess_data(df):
    # Handle missing values
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numerical columns with median
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute categorical columns with the most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Set target as 'Sleep Disorder'
    y = df['Sleep Disorder']

    # Drop the target column from features
    X = df.drop(columns=['Sleep Disorder'])

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, X_encoded.columns

# 3. Train Random Forest Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Take user input from Streamlit
def user_input_features():
    gender = st.selectbox('Gender', ('Male', 'Female'))
    age = st.slider('Age', 1, 100, 25)
    occupation = st.selectbox('Occupation', ('Student', 'Professional', 'Other'))
    sleep_duration = st.slider('Sleep Duration (hours)', 1, 12, 7)
    quality_of_sleep = st.slider('Quality of Sleep (1-5)', 1, 5, 3)
    physical_activity_level = st.slider('Physical Activity Level (1-5)', 1, 5, 3)
    stress_level = st.slider('Stress Level (1-5)', 1, 5, 3)
    bmi_category = st.selectbox('BMI Category', ('Underweight', 'Normal', 'Overweight', 'Obese'))
    blood_pressure = st.slider('Blood Pressure', 60, 200, 120)
    heart_rate = st.slider('Heart Rate', 40, 200, 70)
    daily_steps = st.slider('Daily Steps', 0, 30000, 5000)

    # User input dictionary
    user_data = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity_level,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps
    }

    # Convert to DataFrame for consistency
    user_input_df = pd.DataFrame(user_data, index=[0])
    return user_input_df

# 5. Preprocess user input to match model features
def preprocess_user_input(user_input, feature_columns):
    # One-hot encode the user input
    user_input_encoded = pd.get_dummies(user_input, drop_first=True)

    # Align the columns of user input to match the columns from the training data
    aligned_user_input = user_input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    return aligned_user_input

# 6. Main function to run the Streamlit app
def main():
    st.title('Sleep Disorder Prediction App')

    # Load the dataset
    df = load_data()
    st.write("Dataset Overview:", df.head())

    # Get user input
    user_input = user_input_features()
    st.write("User Input:", user_input)

    # Preprocess data and split into train/test sets
    X_train, X_test, y_train, y_test, feature_columns = preprocess_data(df)

    # Train the Random Forest model
    model = train_model(X_train, y_train)

    # Preprocess user input to match model features
    input_encoded = preprocess_user_input(user_input, feature_columns)

    # Make prediction
    prediction = model.predict(input_encoded)

    # Display prediction
    st.write(f"Prediction: {'Sleep Disorder Detected' if prediction[0] == 1 else 'No Sleep Disorder Detected'}")

    # Test model accuracy 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
