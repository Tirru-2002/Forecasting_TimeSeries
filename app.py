import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

# Load the Holt-Winters model from the pickle file
with open('hwe_model_mul_add.pkl', 'rb') as f:
    hwe_model_mul_add = pickle.load(f)
# Load the dataset
a1 = pd.read_csv('D:\DATA_SCIENCE_COURSE\PROJECTS\PROJECT-3(GOLD)\Gold_data.csv', parse_dates=['date'])  # Replace 'your_dataset.csv' with your actual dataset path
a1.set_index('date', inplace=True)
a1['year'] = a1.index.year  # Create 'year' column based on the datetime index

# Set up Streamlit UI
st.title('Gold Price Analysis')
# Navigation bar
nav_option = st.sidebar.selectbox('Navigation', ['Forecasting', 'Visualize Time Series', 'Display Frequencies', 'Separate Plots by Year', 'Highest Price Each Year', 'Price Distribution'])

# Button click actions
if nav_option == 'Forecasting':
    st.header('Forecasting Gold Prices for Next 30 Days')
    st.write('This plot shows the forecasted gold prices for the next 30 days using the Holt-Winters Exponential Smoothing model. The model was trained on the historical data and used to predict future prices. The plot displays the original data and the forecasted values with a dashed orange line. This visualization can help in understanding the predicted trend of gold prices in the near future.')
    # Forecast the next 30 time periods
    future_data = hwe_model_mul_add.forecast(steps=30)
    # Plotting the original data and the forecasted values
    plt.figure(figsize=(12, 6))
    # plt.plot(a1.index[2152:2183], a1['price'].iloc[2152:2183], label='Original Prices')
    plt.plot(future_data.index, future_data, label='Forecasted Prices', linestyle='dashed', color='orange')
    plt.title('Holt-Winters Exponential Smoothing Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

elif nav_option == 'Visualize Time Series':
    st.header('Visualizing Time Series Data')
    st.write('This plot displays the original time series data of gold prices over time. It provides a visual representation of the price fluctuations and can help identify patterns, trends, and potential seasonality in the data.')
    # Plot the time series data
    plt.figure(figsize=(10, 6))
    plt.plot(a1['price'])
    plt.title('Original Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    st.pyplot(plt)

elif nav_option == 'Display Frequencies':
    st.header('Displaying Frequencies of Prices')
    st.write('This plot shows the density estimation of gold prices using kernel density estimation (KDE). It separates the data into two periods: 2016 to 2019 (green) and 2019 to 2021 (red). This plot can help identify the distribution of prices and any potential shifts or changes in the distribution over time.')
    # Plotting price density
    plt.figure(figsize=(20, 10))
    sns.kdeplot(a1['price'].iloc[:1096], color='green', label='Gold Prices from 2016 to 2019')
    sns.kdeplot(a1['price'].iloc[1096:2183], color='red', label='Gold Prices from 2019 to 2021')
    plt.title('Price Density')
    plt.legend()
    st.pyplot(plt)

elif nav_option == 'Separate Plots by Year':
    st.header('Separate Plots for Each Year')
    st.write('This visualization creates separate subplots for each year in the dataset. Each subplot displays the gold price data for that year, with a 10-day rolling mean line overlaid. This allows for easy comparison of price patterns and trends across different years.')
    # Create separate plots for each year
    years = a1.index.year.unique()
    fig, axes = plt.subplots(nrows=len(years), ncols=1, figsize=(10, 6 * len(years)))
    for i, year in enumerate(years):
        year_data = a1[a1.index.year == year]
        if len(years) == 1:
            ax = axes
        else:
            ax = axes[i]
        year_data["price"].rolling(10).mean().plot(ax=ax)
        ax.set_title(f"Year {year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
    plt.tight_layout()
    st.pyplot(fig)

elif nav_option == 'Highest Price Each Year':
    st.header('Highest Price Recorded Each Year')
    st.write('This bar plot shows the highest price recorded for gold in each year of the dataset. It provides a clear visual representation of the maximum price reached in each year, allowing for quick identification of years with exceptionally high gold prices.')
   # Get the highest price for each year
    highest_prices = a1.groupby('year')['price'].max()
    # Create a bar plot
    fig = go.Figure(data=go.Bar(x=highest_prices.index, y=highest_prices.values))
    fig.update_layout(title_text='Highest Price Recorded Each Year', xaxis_title='Year', yaxis_title='Price')
    st.plotly_chart(fig)

elif nav_option == 'Price Distribution':
    st.header('Distribution of Gold Prices')
    st.write('This plot displays a histogram and kernel density estimation (KDE) curve of the gold price distribution. It helps understand the shape of the price distribution, identify any potential skewness or multi-modality, and get a sense of the range and concentration of prices.')
    # Histogram - Distribution of prices
    plt.figure(figsize=(8, 6))
    sns.histplot(a1['price'], kde=True, bins=12)
    plt.title('Distribution of Gold Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    st.write('This line plot shows the average gold price for each month, revealing potential seasonal patterns or fluctuations in prices throughout the year.')
    # Monthly Seasonality plot
    plt.figure(figsize=(12, 6))
    a1['price'].groupby(a1.index.month).mean().plot(kind='line', color='green')
    plt.title('Seasonality of Gold Prices Over Months')
    plt.xlabel('Month')
    plt.ylabel('Average Price')
    st.pyplot(plt)

    st.write('Similar to the monthly seasonality plot, this line plot displays the average gold price for each day of the month, highlighting any daily patterns or variations in prices.')
    # Daily Seasonality plot
    plt.figure(figsize=(12, 6))
    a1['price'].groupby(a1.index.day).mean().plot(kind='line', color='red')
    plt.title('Seasonality of Daily Gold Prices')
    plt.xlabel('Day')
    plt.ylabel('Average Price')
    st.pyplot(plt)
