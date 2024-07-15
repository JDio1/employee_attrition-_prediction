# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
@st.cache_resource
def load_data():
    df = pd.read_csv('data/cleaned_hr_data.csv')
    return df

df = load_data()

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Transform boolean to binary labels
df['Attrition_Yes'] = df['Attrition_Yes'].apply(lambda x: 1 if x else 0)


# Separate features and target variable
X = df.drop('Attrition_Yes', axis=1)
y = df['Attrition_Yes']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Preprocessing pipeline for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train the model
# (allow_output_mutation=True)
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Streamlit interface
st.title('HR Analytics and Employee Attrition Prediction')

# Display the dataset
st.write("## Dataset")
st.write(df)

# Sidebar for user inputs
st.sidebar.header('User Input Features')

# Collect user inputs
def user_input_features():
    age = st.sidebar.slider('Age', int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))
    monthly_income = st.sidebar.slider('Monthly Income', int(df.MonthlyIncome.min()), int(df.MonthlyIncome.max()), int(df.MonthlyIncome.mean()))
    years_at_company = st.sidebar.slider('Years at Company', int(df.YearsAtCompany.min()), int(df.YearsAtCompany.max()), int(df.YearsAtCompany.mean()))
    
    # Create a dictionary with default values
    data = {col: [0] for col in X.columns}
    
    # Update the dictionary with user input values
    data.update({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'YearsAtCompany': [years_at_company]
    })
    
    # Convert to DataFrame
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Preprocess user input
input_scaled = preprocessor.transform(input_df)

# Make predictions
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.write("## Prediction")
st.write('Attrition:', 'Yes' if prediction[0] else 'No')
st.write('Prediction Probability:', prediction_proba)

# Display additional analysis
st.write("## Additional Analysis")
st.write('Classification Report:')
st.text(classification_report(y_test, model.predict(X_test)))

# Display confusion matrix
st.write("## Confusion Matrix")
conf_matrix = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(conf_matrix, annot=True, fmt="d")
st.pyplot(plt)

st.write("## Exploratory Data Analysis")

# Select feature for histogram
hist_feature = st.selectbox('Select feature for histogram:', df.columns)
st.write(f'Histogram of {hist_feature}')
# Create histogram plot using Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(df[hist_feature], kde=True)
st.pyplot(plt.gcf())  # Display the plot

# Clear the figure for the next plot
plt.clf()

# Select feature for box plot
box_feature = st.selectbox('Select feature for box plot:', df.columns)
st.write(f'Box plot of {box_feature}')

# Create box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x=df[box_feature])
st.pyplot(plt.gcf())  # Display the plot

# Clear the figure to avoid overlap in plots
plt.clf()


# st.pyplot(plt)

# # Select feature for box plot
# box_feature = st.selectbox('Select feature for box plot:', df.columns)
# st.write(f'Box plot of {box_feature}')
# sns.boxplot(x=df[box_feature])
# st.pyplot(plt)






