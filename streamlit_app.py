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


st.title("Finance Adviser")

# this is subtitle 
st.subheader('Raw Data')

# The URL of the CSV file to be read into a DataFrame
csv_url = "./cleaned_data.csv"

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

# Encode 'Occupation' and 'City_Tier' columns
df['Occupation_encode'] = LabelEncoder().fit_transform(df['Occupation'])
df['City_Tier_encode'] = LabelEncoder().fit_transform(df['City_Tier'])


# Transform the variables using Box-Cox transformation

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

X = df.drop(['Occupation', 'City_Tier', 'Income', 'Rent', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Desired_Savings_Percentage', 
             'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare', 'Potential_Savings_Education', 'Potential_Savings_Miscellaneous',
            'Income_transform', 'Rent_transform', 'Insurance_transform', 'Groceries_transform', 'Transport_transform', 'Eating_Out_transform', 'Entertainment_transform', 'Utilities_transform', 'Healthcare_transform', 'Desired_Savings_Percentage',
             'Potential_Savings_Groceries_transform', 'Potential_Savings_Transport_transform', 'Potential_Savings_Eating_Out_transform', 'Potential_Savings_Entertainment_transform', 'Potential_Savings_Utilities_transform', 'Potential_Savings_Healthcare_transform', 'Potential_Savings_Miscellaneous_transform'], axis=1)
y = df['Income_transform']

# Split the dataset into X_train, X_test, y_train, and y_test, 20% of the data for testing
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
# Income = st.slider('Income (in 1000₹)', min_value=df['Income_transform'].min(), max_value=df['Income_transform'].max(), value=df['Income_transform'].mean())
Age = st.slider('Age', min_value=df['Age'].min(), max_value=df['Age'].max(), value=int(df['Age'].mode()))
Dependents = st.slider('Dependents', min_value=df['Dependents'].min(), max_value=df['Dependents'].max(), value=int(df['Dependents'].mode()))
Occupation = st.selectbox('Occupation', ['Professional', 'Student', 'Self_Employed', 'Retired'])
City_Tier = st.selectbox('City_Tier', ['Tier_1', 'Tier_2', 'Tier_3'])
Rent = st.slider('Rent (in 1000₹)', min_value=df['Rent_transform'].min(), max_value=df['Rent_transform'].max(), value=df['Rent_transform'].mean())
Loan_Repayment = st.slider('Loan_Repayment', min_value=df['Loan_Repayment'].min(), max_value=df['Loan_Repayment'].max(), value=df['Loan_Repayment'].mean())
Insurance = st.slider('Insurance (in 1000₹)', min_value=df['Insurance_transform'].min(), max_value=df['Insurance_transform'].max(), value=df['Insurance_transform'].mean())
Groceries = st.slider('Groceries (in 1000₹)', min_value=df['Groceries_transform'].min(), max_value=df['Groceries_transform'].max(), value=df['Groceries_transform'].mean())
Transport = st.slider('Transport (in 1000₹)', min_value=df['Transport_transform'].min(), max_value=df['Transport_transform'].max(), value=df['Transport_transform'].mean())
Eating_Out = st.slider('Eating_Out (in 1000₹)', min_value=df['Eating_Out_transform'].min(), max_value=df['Eating_Out_transform'].max(), value=df['Eating_Out_transform'].mean())
Entertainment = st.slider('Entertainment (in 1000₹)', min_value=df['Entertainment_transform'].min(), max_value=df['Entertainment_transform'].max(), value=df['Entertainment_transform'].mean())
Utilities = st.slider('Utilities (in 1000₹)', min_value=df['Utilities_transform'].min(), max_value=df['Utilities_transform'].max(), value=df['Utilities_transform'].mean())
Healthcare = st.slider('Healthcare (in 1000₹)', min_value=df['Healthcare_transform'].min(), max_value=df['Healthcare_transform'].max(), value=df['Healthcare_transform'].mean())
Education = st.slider('Education', min_value=df['Education'].min(), max_value=df['Education'].max(), value=df['Education'].mean())
Miscellaneous = st.slider('Miscellaneous', min_value=df['Miscellaneous'].min(), max_value=df['Miscellaneous'].max(), value=df['Miscellaneous'].mean())
Desired_Savings_Pourcentage = st.slider('Desired_Savings_Pourcentage (in 1000₹)', min_value=df['Desired_Savings_Pourcentage_transform'].min(), max_value=df['Desired_Savings_Pourcentage_transform'].max(), value=df['Desired_Savings_Pourcentage_transform'].mean())
Desired_Savings = st.slider('Desired_Savings', min_value=df['Desired_Savings'].min(), max_value=df['Desired_Savings'].max(), value=df['Desired_Savings'].mean())
Disposable_Income = st.slider('Disposable_Income', min_value=df['Disposable_Income'].min(), max_value=df['Disposable_Income'].max(), value=df['Disposable_Income'].mean())
Potential_Savings_Groceries = st.slider('Potential_Savings_Groceries (in 1000₹)', min_value=df['Potential_Savings_Groceries_transform'].min(), max_value=df['Potential_Savings_Groceries_transform'].max(), value=df['Potential_Savings_Groceries_transform'].mean())
Potential_Savings_Transport = st.slider('Potential_Savings_Transport (in 1000₹)', min_value=df['Potential_Savings_Transport_transform'].min(), max_value=df['Potential_Savings_Transport_transform'].max(), value=df['Potential_Savings_Transport_transform'].mean())
Potential_Savings_Eating_Out = st.slider('Potential_Savings_Eating_Out (in 1000₹)', min_value=df['Potential_Savings_Eating_Out_transform'].min(), max_value=df['Potential_Savings_Eating_Out_transform'].max(), value=df['Potential_Savings_Eating_Out_transform'].mean())
Potential_Savings_Entertainment = st.slider('Potential_Savings_Entertainment (in 1000₹)', min_value=df['Potential_Savings_Entertainment_transform'].min(), max_value=df['Potential_Savings_Entertainment_transform'].max(), value=df['Potential_Savings_Entertainment_transform'].mean())
Potential_Savings_Utilities = st.slider('Potential_Savings_Utilities (in 1000₹)', min_value=df['Potential_Savings_Utilities_transform'].min(), max_value=df['Potential_Savings_Utilities_transform'].max(), value=df['Potential_Savings_Utilities_transform'].mean())
Potential_Savings_Healthcare = st.slider('Potential_Savings_Healthcare (in 1000₹)', min_value=df['Potential_Savings_Healthcare_transform'].min(), max_value=df['Potential_Savings_Healthcare_transform'].max(), value=df['Potential_Savings_Healthcare_transform'].mean())
Potential_Savings_Education = st.slider('Potential_Savings_Education', min_value=df['Potential_Savings_Education'].min(), max_value=df['Potential_Savings_Education'].max(), value=df['Potential_Savings_Education'].mean())
Potential_Savings_Miscellaneous = st.slider('Potential_Savings_Miscellaneous (in 1000₹)', min_value=df['Potential_Savings_Miscellaneous_transform'].min(), max_value=df['Potential_Savings_Miscellaneous_transform'].max(), value=df['Potential_Savings_Miscellaneous_transform'].mean())


# Encode categorical variables for user input
# sex_encode = 1 if sex == 'female' else 0
Occupation_encode = ['Professional', 'Student', 'Self_Employed', 'Retired'].index(Occupation)
City_Tier_encode = ['Tier_1', 'Tier_2', 'Tier_3'].index(City_Tier)

# Total stuff
# total_potential_savings = (Potential_Savings_Groceries + Potential_Savings_Transport + Potential_Savings_Eating_Out + Potential_Savings_Entertainment + Potential_Savings_Utilities + Potential_Savings_Healthcare + Potential_Savings_Education +  Potential_Savings_Miscellaneous)
# total_expenses = (Loan_Repayment + Insurance + Groceries + Transport + Eating_Out + Entertainment + Utilities + Healthcare + Education + Miscellaneous)
# st.write(total_potential_savings)

# Predict charges
predicted_Income_transformed = linear_model.predict([[Age, Dependents, Occupation_encode, City_Tier_encode, Rent, Desired_Savings_Pourcentage, Desired_Savings, Disposable_Income]])

# Reverse the Box-Cox transformation, no one knows, search later
predicted_Income = inv_boxcox(predicted_Income_transformed, lambda_value)

st.write('updated')

# Display prediction
st.write('Predicted Income:', round(predicted_Income[0], 0))
