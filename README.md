# HR Analytics and Employee Attrition Prediction

## Problem Statement

Employee attrition is a significant challenge for companies as it leads to increased costs and disruption. Identifying the key factors influencing attrition and predicting which employees are at risk of leaving can help HR departments take proactive measures to retain valuable talent.

## Project Overview

This project aims to analyze employee data to identify factors that contribute to employee attrition and develop predictive models to forecast which employees are likely to leave the company. The purpose of this project is to provide insights that can help reduce employee turnover and improve retention strategies.

## Project Objectives
 The key objectives are:

- Perform exploratory data analysis (EDA) to understand the dataset and identify important features.
- Develop and evaluate machine learning models to predict employee attrition.
- Visualize the findings using Tableau Public to provide actionable insights.

## Methodologies

1. **Data Collection and Cleaning**
   - The used was “HR Employee Attrition” dataset from Kaggle, which includes various HR-related metrics like employee demographics, job roles, performance ratings, engagement scores, and attrition status.
   - Loaded and cleaned the HR dataset to handle missing values and ensure data quality.
   - Converted categorical variables into numerical representations using one-hot encoding.

2. **Exploratory Data Analysis (EDA)**
   - Conducted univariate, bivariate, and multivariate analysis to understand data distributions and relationships between features.
   - Visualized key patterns and insights using plots.

3. **Predictive Modeling**
   - Split the data into training and test sets.
   - Trained multiple machine learning models, including Random Forest and Logistic Regression.
   - Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.

4. **Data Visualization and Dashboard**
   - Created interactive dashboards in Tableau Public to visualize attrition trends and key features influencing attrition.
    !["HR Attrition Dashboard(Made with Tableu)"](https://github.com/JDio1/employee_attrition-_prediction/blob/main/HR%20Attrition%20Dashboard.png)

## Technologies Used

- **Python**: Data analysis and machine learning
- **Jupyter Notebook**: Development environment
- **Pandas, Numpy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **Matplotlib, Seaborn**: Data visualization
- **Tableau Public**: Interactive dashboards and data visualization

## Summary of Results

- **Exploratory Data Analysis (EDA)**: Identified key factors such as age, monthly income, and job satisfaction that correlate with employee attrition.
- **Predictive Modeling**: Developed models, including Random Forest and Logistic Regression, achieving an accuracy of 86% in predicting employee attrition.
- **Data Visualization**: Created an interactive Tableau dashboard to visualize attrition trends and key features influencing attrition.
These findings and tools provide actionable insights for HR departments to develop effective retention strategies and reduce employee turnover.


## Reproducing the Analysis

### Prerequisites

- Python 3.7 or above
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Tableau Public for data visualization

### Instructions

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/hr-analytics-attrition.git
   cd hr-analytics-attrition

2. **Install Dependencies**
        pip install -r requirements.txt

3. **Run Jupyter Notebooks**
   - **preprocess.ipyng**:
        Preprocesses and cleans the data from the dataset
   - **EDA.ipynb**: 
        Performs Exploratory Data Analysis (EDA) on the cleaned data
   - **Predictive modeling.ipynb**:
        Predictive models are developed, evaluated, trained, and compared to predict employee turnover

## Author
- Justin Uto-Dieu