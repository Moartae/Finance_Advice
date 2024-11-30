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

st.title("Finance Adviser")

# Subtitle 
st.subheader('Raw Data')

# The URL of the CSV file to be read into a DataFrame
csv_url = "./cleaned_data_no_negatives.csv"

# Reading the CSV data from the specified URL into a DataFrame named 'df'
df = pd.read_csv(csv_url)

# Display the dataset
st.write(df)

# Remove duplicate row from dataset
df.drop_duplicates(keep='first', inplace=True)

# ... [rest of the previous data visualization and transformation code remains the same]
st.write('### Display Numerical Plots')

# Select box to choose which feature to plot
feature_to_plot = st.selectbox('Select a numerical feature to plot', 
                               ['Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 
                                'Eating_Out', 'Entertainment', 'Utilities', 'Healhcare', 'Education', 'Miscellaneous', 
                                'Desired_Saving_Pourcentage', 'Desired_Savings', 'Potential_Savings_Groceries', 'Potential_Savings_Transport', 
                                'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 
                                'Potential_Savings_Healthcare', 'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'])

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
# ... [previous transformations remain the same]
df['Income_transform'], lambda_Income = stats.boxcox(df['Income'])
df['Rent_transform'], lambda_Rent = stats.boxcox(df['Rent'])
# df['Loan_Repayment_transform'], lambda_value = stats.boxcox(df['Loan_Repayment'])
df['Insurance_transform'], lambda_Insurance = stats.boxcox(df['Insurance'])
df['Groceries_transform'], lambda_Groceries = stats.boxcox(df['Groceries'])
df['Transport_transform'], lambda_Transport = stats.boxcox(df['Transport'])
df['Eating_Out_transform'], lambda_Eating_Out = stats.boxcox(df['Eating_Out'])
df['Entertainment_transform'], lambda_Entertainment = stats.boxcox(df['Entertainment'])
df['Utilities_transform'], lambda_Utilities = stats.boxcox(df['Utilities'])
df['Healthcare_transform'], lambda_Healthcare = stats.boxcox(df['Healthcare'])
# df['Education_transform'], lambda_value = stats.boxcox(df['Education'])
# df['Miscellaneous_transform'], lambda_value = stats.boxcox(df['Miscellaneuous'])
df['Desired_Savings_Percentage_transform'], lambda_Desired_Savings_Percentage = stats.boxcox(df['Desired_Savings_Percentage'])
# df['Desired_Savings_transform'], lambda_value = stats.boxcox(df['Desired_Savings'])
# df['Disposable_Income_transform'], lambda_value = stats.boxcox(df['Disposable_Income'])
df['Potential_Savings_Groceries_transform'], lambda_Potential_Savings_Groceries = stats.boxcox(df['Potential_Savings_Groceries'])
df['Potential_Savings_Transport_transform'], lambda_Potential_Savings_Transport = stats.boxcox(df['Potential_Savings_Transport'])
df['Potential_Savings_Eating_Out_transform'], lambda_Potential_Savings_Eating_Out = stats.boxcox(df['Potential_Savings_Eating_Out'])
df['Potential_Savings_Entertainment_transform'], lambda_Potential_Savings_Entertainment = stats.boxcox(df['Potential_Savings_Entertainment'])
df['Potential_Savings_Utilities_transform'], lambda_Potential_Savings_Utilities = stats.boxcox(df['Potential_Savings_Utilities'])
df['Potential_Savings_Healthcare_transform'], lambda_Potential_Savings_Healthcare = stats.boxcox(df['Potential_Savings_Healthcare'])
# df['Potential_Savings_Education_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Education'])
# df['Potential_Savings_Miscellaneous_transform'], lambda_value = stats.boxcox(df['Potential_Savings_Miscellaneous'])

# Define X (features) and y (target)
# Include all transformed and encoded features
X = df[['Age', 'Dependents', 'Occupation_encode', 'City_Tier_encode', 
        'Rent_transform', 'Loan_Repayment', 'Insurance_transform', 
        'Groceries_transform', 'Transport_transform', 'Eating_Out_transform', 
        'Entertainment_transform', 'Utilities_transform', 'Healthcare_transform', 
        'Education', 'Miscellaneous', 'Desired_Savings_Percentage_transform', 
        'Potential_Savings_Groceries_transform', 'Potential_Savings_Transport_transform', 
        'Potential_Savings_Eating_Out_transform', 'Potential_Savings_Entertainment_transform', 
        'Potential_Savings_Utilities_transform', 'Potential_Savings_Healthcare_transform']]

y = df['Income_transform']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Streamlit app for prediction
st.write('## Predict Your Income')

# User input for features
Age = st.slider('Age', min_value=df['Age'].min(), max_value=df['Age'].max(), value=int(df['Age'].mode()))
Dependents = st.slider('Dependents', min_value=df['Dependents'].min(), max_value=df['Dependents'].max(), value=int(df['Dependents'].mode()))
Occupation = st.selectbox('Occupation', ['Professional', 'Student', 'Self_Employed', 'Retired'])
City_Tier = st.selectbox('City_Tier', ['Tier_1', 'Tier_2', 'Tier_3'])
Rent = st.slider('Rent (₹)', min_value=df['Rent'].min(), max_value=df['Rent'].max(), value=df['Rent'].mean())
Loan_Repayment = st.slider('Loan_Repayment (₹)', min_value=df['Loan_Repayment'].min(), max_value=df['Loan_Repayment'].max(), value=df['Loan_Repayment'].mean())
Insurance = st.slider('Insurance (₹)', min_value=df['Insurance'].min(), max_value=df['Insurance'].max(), value=df['Insurance'].mean())
Groceries = st.slider('Groceries (₹)', min_value=df['Groceries'].min(), max_value=df['Groceries'].max(), value=df['Groceries'].mean())
Transport = st.slider('Transport (₹)', min_value=df['Transport'].min(), max_value=df['Transport'].max(), value=df['Transport'].mean())
Eating_Out = st.slider('Eating_Out (₹)', min_value=df['Eating_Out'].min(), max_value=df['Eating_Out'].max(), value=df['Eating_Out'].mean())
Entertainment = st.slider('Entertainment (₹)', min_value=df['Entertainment'].min(), max_value=df['Entertainment'].max(), value=df['Entertainment'].mean())
Utilities = st.slider('Utilities (₹)', min_value=df['Utilities'].min(), max_value=df['Utilities'].max(), value=df['Utilities'].mean())
Healthcare = st.slider('Healthcare (₹)', min_value=df['Healthcare'].min(), max_value=df['Healthcare'].max(), value=df['Healthcare'].mean())
Education = st.slider('Education (₹)', min_value=df['Education'].min(), max_value=df['Education'].max(), value=df['Education'].mean())
Miscellaneous = st.slider('Miscellaneous (₹)', min_value=df['Miscellaneous'].min(), max_value=df['Miscellaneous'].max(), value=df['Miscellaneous'].mean())
Desired_Savings_Percentage = st.slider('Desired_Savings_Percentage (₹)', min_value=df['Desired_Savings_Percentage'].min(), max_value=df['Desired_Savings_Percentage'].max(), value=df['Desired_Savings_Percentage'].mean())
Potential_Savings_Groceries = st.slider('Potential_Savings_Groceries (₹)', min_value=df['Potential_Savings_Groceries'].min(), max_value=df['Potential_Savings_Groceries'].max(), value=df['Potential_Savings_Groceries'].mean())
Potential_Savings_Transport = st.slider('Potential_Savings_Transport (₹)', min_value=df['Potential_Savings_Transport'].min(), max_value=df['Potential_Savings_Transport'].max(), value=df['Potential_Savings_Transport'].mean())
Potential_Savings_Eating_Out = st.slider('Potential_Savings_Eating_Out (₹)', min_value=df['Potential_Savings_Eating_Out'].min(), max_value=df['Potential_Savings_Eating_Out'].max(), value=df['Potential_Savings_Eating_Out'].mean())
Potential_Savings_Entertainment = st.slider('Potential_Savings_Entertainment (₹)', min_value=df['Potential_Savings_Entertainment'].min(), max_value=df['Potential_Savings_Entertainment'].max(), value=df['Potential_Savings_Entertainment'].mean())
Potential_Savings_Utilities = st.slider('Potential_Savings_Utilities (₹)', min_value=df['Potential_Savings_Utilities'].min(), max_value=df['Potential_Savings_Utilities'].max(), value=df['Potential_Savings_Utilities'].mean())
Potential_Savings_Healthcare = st.slider('Potential_Savings_Healthcare (₹)', min_value=df['Potential_Savings_Healthcare'].min(), max_value=df['Potential_Savings_Healthcare'].max(), value=df['Potential_Savings_Healthcare'].mean())

# tranform user input
# df[Rent], lambda_value = stats.boxcox(Rent)[0][0]
Rent_transformed = stats.boxcox([Rent], lambda_Rent)
Insurance_transformed = stats.boxcox([Insurance], lambda_Insurance)
Groceries_transformed = stats.boxcox([Groceries], lambda_Groceries)
Transport_transformed = stats.boxcox([Transport], lambda_Transport)
Eating_Out_transformed = stats.boxcox([Eating_Out], lambda_Eating_Out)
Entertainment_transformed = stats.boxcox([Entertainment], lambda_Entertainment)
Utilities_transformed = stats.boxcox([Utilities], lambda_Utilities)
Healthcare_transformed = stats.boxcox([Healthcare], lambda_Healthcare)
Desired_Savings_Percentage_transformed = stats.boxcox([Desired_Savings_Percentage], lambda_Desired_Savings_Percentage)
Potential_Savings_Groceries_transformed = stats.boxcox([Potential_Savings_Groceries], lambda_Potential_Savings_Groceries)
Potential_Savings_Transport_transformed = stats.boxcox([Potential_Savings_Transport], lambda_Potential_Savings_Transport)
Potential_Savings_Eating_Out_transformed = stats.boxcox([Potential_Savings_Eating_Out], lambda_Potential_Savings_Eating_Out)
Potential_Savings_Entertainment_transformed = stats.boxcox([Potential_Savings_Entertainment], lambda_Potential_Savings_Entertainment)
Potential_Savings_Utilities_transformed = stats.boxcox([Potential_Savings_Utilities], lambda_Potential_Savings_Utilities)
Potential_Savings_Healthcare_transformed = stats.boxcox([Potential_Savings_Healthcare], lambda_Potential_Savings_Healthcare)

# Encode categorical variables for user input
Occupation_encode = ['Professional', 'Student', 'Self_Employed', 'Retired'].index(Occupation)
City_Tier_encode = ['Tier_1', 'Tier_2', 'Tier_3'].index(City_Tier)

# Prepare input for prediction matching the training data
user_input = np.array([[
    Age, Dependents, Occupation_encode, City_Tier_encode, 
    float(Rent_transformed), Loan_Repayment, float(Insurance_transformed), 
    float(Groceries_transformed), float(Transport_transformed), float(Eating_Out_transformed), 
    float(Entertainment_transformed), float(Utilities_transformed), float(Healthcare_transformed), 
    Education, Miscellaneous, float(Desired_Savings_Percentage_transformed), 
    float(Potential_Savings_Groceries_transformed), float(Potential_Savings_Transport_transformed), 
    float(Potential_Savings_Eating_Out_transformed), float(Potential_Savings_Entertainment_transformed), 
    float(Potential_Savings_Utilities_transformed), float(Potential_Savings_Healthcare_transformed)
]]).reshape(1, -1)

# Predict income
predicted_Income_transformed = linear_model.predict(user_input)

# Use inv_boxcox to transform back
# Important: inv_boxcox expects a scalar, so use predicted_Income_transformed[0]
predicted_Income = inv_boxcox(predicted_Income_transformed[0], lambda_Income)

# Display prediction
st.write('# Predicted Income:', round(predicted_Income, 0))
