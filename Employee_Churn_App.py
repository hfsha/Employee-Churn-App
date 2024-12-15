import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing

# --- Load Dataset ---
@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/Shahidatul Hidayah/OneDrive/Documents/SEM 5/PRA/Asg2/HR_comma_sep.csv")
    return data

data = load_data()

# --- Data Preprocessing ---
def preprocess_data(data):
    # Handle missing values
    numeric_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                        'average_montly_hours', 'time_spend_company']
    for feature in numeric_features:
        if data[feature].isnull().sum() > 0:
            data[feature].fillna(data[feature].mean(), inplace=True)

    categorical_features = ['Work_accident', 'promotion_last_5years', 'Departments', 'salary']
    for feature in categorical_features:
        if data[feature].isnull().sum() > 0:
            data[feature].fillna(data[feature].mode()[0], inplace=True)
    
    # Label encoding for categorical features
    le = preprocessing.LabelEncoder()
    data['salary'] = le.fit_transform(data['salary'])
    data['Departments'] = le.fit_transform(data['Departments'])

    return data

data = preprocess_data(data)

# --- Train-Test Split ---
X = data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 
          'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments', 'salary']]
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
@st.cache_resource
def train_best_model(X_train, y_train):
    # Hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
        'alpha': [0.001, 0.01, 0.1],
        'max_iter': [500],
        'learning_rate': ['constant', 'adaptive']
    }
    grid_search = GridSearchCV(MLPClassifier(solver='adam', random_state=5), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model

best_model = train_best_model(X_train, y_train)

# --- App Interface ---
st.title("Employee Churn Prediction")
st.write("This application predicts whether an employee is likely to leave the company.")

# Sidebar Inputs
st.sidebar.header("Employee Input Features")
def user_input_features():
    satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    last_evaluation = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5)
    number_project = st.sidebar.slider('Number of Projects', 1, 10, 4)
    average_montly_hours = st.sidebar.slider('Average Monthly Hours', 50, 300, 150)
    time_spend_company = st.sidebar.slider('Time Spent in Company (years)', 1, 10, 3)
    Work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
    promotion_last_5years = st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1])
    Departments = st.sidebar.slider('Department (Encoded)', 0, 9, 2)
    salary = st.sidebar.slider('Salary Level (Encoded)', 0, 2, 1)

    data = {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': Work_accident,
        'promotion_last_5years': promotion_last_5years,
        'Departments': Departments,
        'salary': salary
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction ---
if st.button("Predict Churn"):
    prediction = best_model.predict(input_df)
    prediction_proba = best_model.predict_proba(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with the company.")

    st.subheader("Prediction Probability")
    st.write(f"Stay Probability: {prediction_proba[0][0] * 100:.2f}%")
    st.write(f"Leave Probability: {prediction_proba[0][1] * 100:.2f}%")

# --- Visualizations ---
st.subheader("Dataset Overview")
st.dataframe(data.head())

st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Loss Curve of the Best Model")
plt.plot(best_model.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Cost")
st.pyplot(plt)

st.sidebar.write("\n---\nDeveloped with Streamlit")