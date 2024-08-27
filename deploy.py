import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


model = load_model('stock prediction.kreas')

st.title('Stock Price Predictions')

st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

def main():
    option = st.sidebar.selectbox('Make a choice', ['EDA','Visualize','Recent Data', 'Predict'])
    if option == 'EDA':
        perform_eda(data)
    elif option == 'Visualize':
        visualization()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

@ st.cache_data
def load_data():
    file_path = 'RELIANCE_STOCK (1).csv'  # Update this to the correct path
    data = pd.read_csv(file_path)
    return data


# Set the date range
min_date = datetime.date(2000, 1, 3)
max_date = datetime.date(2022, 12, 30)

# Default values for duration
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=8030)
before = today - datetime.timedelta(days=duration)

# Ensure the default 'before' does not go beyond 'min_date'
if before < min_date:
    before = min_date

# Ensure the default 'today' does not go beyond 'max_date'
if today > max_date:
    today = max_date

# Date input fields with the specified date range
start_date = st.sidebar.date_input('Start Date', value=before, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End Date', value=today, min_value=min_date, max_value=max_date)

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
        load_data()
    else:
        st.sidebar.error('Error: End date must fall after start date')


data = load_data()
scaler = StandardScaler()

def perform_eda(data):
    st.write("### Exploratory Data Analysis")
    st.write("Here is a preview of the dataset:")
    st.dataframe(data)

    st.write("#### Summary Statistics")
    st.write(data.describe())

    st.write("#### Missing Values")
    st.write(data.isnull().sum())
    
    # Additional EDA features can be added here
    st.write("#### Correlation Matrix")
    st.write(data.corr())


def visualization():
    st.header('Visualization')
    option = st.selectbox(
    "Select the plot you want to see",
    ("Distribution of Numerical Columns", 
     "Stock Price with Moving Averages") )

    # Plotting based on user selection
    if option == "Distribution of Numerical Columns":
        st.write("### Distribution of Numerical Columns")
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            st.write(f"**Distribution of {col}**")
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            st.pyplot(plt)
            plt.close()  # Clear the figure after displaying

    elif option == "Stock Price with Moving Averages":
        # Moving average calculation
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        st.write("### Stock Price with Moving Averages")
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.plot(data.index, data['MA50'], label='50-Day Moving Average')
        plt.plot(data.index, data['MA200'], label='200-Day Moving Average')
        plt.title('Reliance Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(plt)
        plt.clf()



def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

data = pd.DataFrame({
    'Close': np.random.randn(100),  # Example data
    'Feature1': np.random.randn(100),  # Example feature
})


X = data[['Feature1']]  # Example feature
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_predict(model_name, X_train, y_train, X_test):
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'SVM Model':
        model = SVR()
    elif model_name == 'DecisionTree Regression':
        model = DecisionTreeRegressor()
    elif model_name == 'ARIMA':
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()
        return model_fit.forecast(steps=len(X_test))
    else:
        raise ValueError("Model not recognized")

    model.fit(X_train, y_train)
    return model.predict(X_test)




def predict():
    model_name = st.radio('Choose a model', ['LinearRegression', 'ARIMA', 'SVM Model', 'DecisionTree Regression'])
    
    if model_name == 'ARIMA':
        st.write("ARIMA requires time-series data. Ensure your data is appropriately formatted.")
        predictions = train_and_predict(model_name, None, y_train, X_test)
    else:
        predictions = train_and_predict(model_name, X_train, y_train, X_test)
    
    st.write(f"### Predictions using {model_name}")
    st.write(predictions)

    # Calculate and display the mean squared error for models that are not ARIMA
    if model_name != 'ARIMA':
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Mean Squared Error: {mse:.2f}")


if __name__ == '__main__':
    main()
