import streamlit as st

import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px

import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot


st.title("Regression")

st.subheader('Raw Data')

csv_url = "./data2.csv"

# Reading the CSV data from the specified URL into a DataFrame named 'df'
df = pd.read_csv(csv_url)

# Display the dataset
st.write(df)

# Remove duplicate row from dataset
df.drop_duplicates(keep='first', inplace=True)

st.write('### Display Numerical Plots')

# Select box to choose which feature to plot
feature_to_plot = st.selectbox('Select a numerical feature to plot', ['Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healhcare', 'Education', 'Miscellaneous', 'Desired_Saving_Pourcentage', 'Desired_Savings', 'Disposable_Income', 'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare', 'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'])

# Plot the selected feature
if feature_to_plot:
    st.write(f'Distribution of {feature_to_plot}:')
    fig = plt.figure(figsize=(10, 6))
    plt.hist(df[feature_to_plot], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(feature_to_plot)
    plt.ylabel('Count')
    st.pyplot(fig)

st.write('### Display Categorical Plots')

# Select box to choose which feature to plot
feature_to_plot = st.selectbox('Select a feature to plot', ['Occupation', 'City_Tier',])

# Plot the selected categorical feature
if feature_to_plot:
    st.write(f'Distribution of {feature_to_plot}:')
    bar_chart = st.bar_chart(df[feature_to_plot].value_counts())

st.write('### Display Relationships')

# Create dropdown menus for user selection
x_variable = st.selectbox('Select x-axis variable:', df.columns)
y_variable = st.selectbox('Select y-axis variable:', df.columns)
color_variable = st.selectbox('Select color variable:', df.columns)
size_variable = st.selectbox('Select size variable:', df.columns)

# Scatter plot with Plotly Express
fig = px.scatter(df, x=x_variable, y=y_variable, color=color_variable, size=size_variable, hover_data=[color_variable])

# Display the plot
st.plotly_chart(fig)

# Encode 'sex', 'smoker', and 'region' columns
df['Occupation_encode'] = LabelEncoder().fit_transform(df['Occupation'])
df['City_Tier_encode'] = LabelEncoder().fit_transform(df['City_Tier'])


# Transform the 'charges' variable using Box-Cox transformation
df['Income_transform'], lambda_value = stats.boxcox(df['Income'])
df['Rent_transform'], lambda_value = stats.boxcox(df['Rent'])
# df['Loan_Repayment_transform'], lambda_value = stats.boxcox(df['Loan_Repayment'])
df['Insurance_transform'], lambda_value = stats.boxcox(df['Insurance'])
df['Groceries_transform'], lambda_value = stats.boxcox(df['Groceries'])
df['Transport_transform'], lambda_value = stats.boxcox(df['Transport'])
df['Eating_Out_transform'], lambda_value = stats.boxcox(df['Eating_Out'])
df['Entertainment_transform'], lambda_value = stats.boxcox(df['Entertainment'])
df['Utilities_transform'], lambda_value = stats.boxcox(df['Utilities'])
df['Healthcare_transform'], lambda_value = stats.boxcox(df['Healthcare'])
# df['Education_transform'], lambda_value = stats.boxcox(df['Education'])
# df['Miscellaneous_transform'], lambda_value = stats.boxcox(df['Miscellaneuous'])
df['Desired_Savings_Pourcentage_transform'], lambda_value = stats.boxcox(df['Desired_Savings_Percentage'])
# df['Desired_Savings_transform'], lambda_value = stats.boxcox(df['Desired_Savings'])
# df['Disposable_Income_transform'], lambda_value = stats.boxcox(df['Disposable_Income'])
df['Potential_Savings_Groceries_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Groceries'])
df['Potential_Savings_Transport_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Transport'])
df['Potential_Savings_Eating_Out_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Eating_Out'])
df['Potential_Savings_Entertainment_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Entertainment'])
df['Potential_Savings_Utilities_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Utilities'])
df['Potential_Savings_Healthcare_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Healthcare'])
# df['Potential_Savings_Education_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Education'])
df['Potential_Savings_Miscellaneous_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Miscellaneous'])


# Define X (features) and y (target) and remove duplicate features that will not be used in the model
X = df.drop(['Occupation', 'City_Tier', 'Income', 'Rent', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Desired_Savings_Percentage', 'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare', 'Potential_Savings_Education', 'Potential_Savings_Miscellaneous',
            'Income_transform', 'Rent_transform', 'Insurance_transform', 'Groceries_transform', 'Transport_transform', 'Eating_Out_transform', 'Entertainment_transform', 'Utilities_transform', 'Healthcare_transform', 'Desired_Savings_Percentage', 'Potential_Savings_Groceries_transform', 'Potential_Savings_Transport_transform', 'Potential_Savings_Eating_Out_transform', 'Potential_Savings_Entertainment_transform', 'Potential_Savings_Utilities_transform', 'Potential_Savings_Healthcare_transform', 'Potential_Savings_Miscellaneous_transform'], axis=1)
y = df['Income_transform']
y = df['Rent_transform']
y = df['Insurance_transform']
y = df['Groceries_transform']
y = df['Transport_transform']
y = df['Eating_Out_transform']
y = df['Entertainment_transform']
y = df['Utilities_transform']
y = df['Healthcare_transform']
y = df['Desired_Savings_Pourcentage_transform']
y = df['Potential_Savings_Groceries_transform']
y = df['Potential_Savings_Transport_transform']
y = df['Potential_Savings_Eating_Out_transform']
y = df['Potential_Savings_Entertainment_transform']
y = df['Potential_Savings_Utilities_transform']
y = df['Potential_Savings_Healthcare_transform']
y = df['Potential_Savings_Miscellaneous_transform']
#  ' 'Desired_Savings_Percentage', 'Potential_Savings_Groceries_transform', 'Potential_Savings_Transport_transform', 'Potential_Savings_Eating_Out_transform', 'Potential_Savings_Entertainment_transform', 'Potential_Savings_Utilities_transform', 'Potential_Savings_Healthcare_transform', 'Potential_Savings_Miscellaneous_transform']

# Split the dataset into X_train, X_test, y_train, and y_test, 10% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate a linear regression model
linear_model = LinearRegression()

# Fit the model using the training data
linear_model.fit(X_train, y_train)

# For each record in the test set, predict the y value (transformed value of charges)
# The predicted values are stored in the y_pred array
y_pred = linear_model.predict(X_test)

# Create Streamlit app
st.write('## Predict Your Own Charges')

# User input for features
Income = st.slider('Income (in 1000$)', min_value=df['Income_transform'].min(), max_value=df['Income_transform'].max(), value=df['Income_transform'].mean())
Age = st.slider('Age', min_value=df['Age'].min(), max_value=df['Age'].max(), value=int(df['Age'].mode()))
Dependents = st.slider('Dependents', min_value=df['Dependents'].min(), max_value=df['Dependents'].max(), value=int(df['Dependents'].mode()))
Occupation = st.selectbox('Occupation', ['Professional', 'Student', 'Self_Employed', 'Retired'])
City_Tier = st.selectbox('City_Tier', ['Tier_1', 'Tier_2', 'Tier_3'])
Rent = st.slider('Rent (in 1000$)', min_value=df['Rent_transform'].min(), max_value=df['Rent_transform'].max(), value=df['Rent_transform'].mean())
Loan_Repayment = st.slider('Loan_Repayment', min_value=df['Loan_Repayment'].min(), max_value=df['Loan_Repayment'].max(), value=df['Loan_Repayment'].mean())
Insurance = st.slider('Insurance (in 1000$)', min_value=df['Insurance_transform'].min(), max_value=df['Insurance_transform'].max(), value=df['Insurance_transform'].mean())

