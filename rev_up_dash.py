import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import csv
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st. set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

def get_urls():
    base_url = "https://www.smev.in/"
    try:
        # Send a GET request to the base URL
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the links to the fiscal year pages
        links = soup.find_all('a', href=True)
        
        # Extract URLs for fiscal years
        urls = [link['href'] for link in links if link['href'].split('/')[-1].startswith('fy-')]

        return urls

    except Exception as e:
        st.error(f"An error occurred while retrieving URLs: {e}")
        return []

def extract_ev_2w_sales(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table containing the EV 2W sales data
        table = soup.find('table')

        if table:
            # Extract table rows
            rows = table.find_all('tr')

            # Extracting data from table rows
            data = []
            for row in rows:
                cells = row.find_all(['th', 'td'])
                data.append([cell.get_text(strip=True) for cell in cells])

            return data
        else:
            st.warning("No table found on the webpage.")
            return None

    except Exception as e:
        st.error(f"An error occurred while extracting data from {url}: {e}")
        return None

def scrape_ev_sales(csv_file):
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        existing_years = set()
        # Read existing years from the CSV file
        with open(csv_file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                existing_years.add(row[0])

        st.write("Existing years:", existing_years)
    else:
        existing_years = set()

    # Retrieve URLs for each fiscal year
    urls = get_urls()

    # Open the CSV file in append mode
    with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Iterate over each URL and extract data
        for url in urls:
            year = url.split('/')[-1]
            
            # Check if data for the current year already exists
            if year in existing_years:
                st.warning(f"Data for {year} already exists in the CSV file. Skipping extraction.")
                continue

            st.write(f"Processing URL: {url}")
            data = extract_ev_2w_sales(url)

            if data:
                # Write the extracted data into the CSV file
                for row in data[1:]:  # Skip the first row (headers)
                    writer.writerow([year] + row)

                st.success(f"Data for {year} extracted and added to the CSV file")
            else:
                st.error(f"No data extracted from {url}")

    st.success("Process completed.")

#------------------------------------------------------------------------------------------------------------------------------
def plot_sales_trend2(df, total_sales=True, selected_company=None):#plotly individual trendlines
    sales_data = df.copy()

    # Calculate total sales for each year
    total_sales_by_year = sales_data.groupby('Year')['Total'].sum().reset_index()

    # Check if specific company is selected and data for the last year is available
    if selected_company:
        company_data = sales_data[sales_data['Maker'].str.lower().str.contains(selected_company.lower())]
        if company_data.empty or total_sales_by_year.empty or total_sales_by_year['Year'].iloc[-1] != int(pd.Timestamp.now().year)-1:
            st.error(f"No data found for {selected_company} for the last year.")
            return
        else:
            total_sales_by_year = company_data.groupby('Year')['Total'].sum().reset_index()

    if total_sales_by_year.empty:
        st.error("No data available.")
        return

    # Fit polynomial regression model
    degree = 2
    X = total_sales_by_year['Year'].values.reshape(-1, 1)
    y = total_sales_by_year['Total'].values
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predictions for the next year
    future_year = total_sales_by_year['Year'].max() + 1
    future_year_poly = poly_features.transform([[future_year]])
    sales_pred = model.predict(future_year_poly)[0]

    # Calculate confidence interval for prediction
    residuals = y - model.predict(X_poly)
    confidence_interval = 1.96 * np.std(residuals)

    # Plot total sales trend
    trace_actual = go.Scatter(x=total_sales_by_year['Year'], y=total_sales_by_year['Total'], mode='markers', name='Actual Sales')
    trace_trend = go.Scatter(x=total_sales_by_year['Year'], y=model.predict(X_poly), mode='lines', name='Trendline')
    layout = go.Layout(title='Total Sales Trend', xaxis=dict(title='Year'), yaxis=dict(title='Total Sales'))
    fig = go.Figure(data=[trace_actual, trace_trend], layout=layout)

    # Add prediction for the next year with confidence interval
    #if selected_company:
    fig.add_trace(go.Scatter(x=[future_year], y=[sales_pred], mode='markers', name='Predicted Sales'))
    fig.add_shape(type="line",
                    x0=future_year, y0=sales_pred-confidence_interval,
                    x1=future_year, y1=sales_pred+confidence_interval,
                    line=dict(color="red", width=2, dash="dot"), name='Confidence Interval')

    # Show plot
    st.plotly_chart(fig, width=100, height=100)
    if not selected_company:
        st.success(f"Total sales prediction for {future_year}: {sales_pred:.2f}")
    else:
        st.success(f"Total sales prediction of {selected_company} for {future_year}: {sales_pred:.2f}")
    

def plot_sales_time_series(df, selected_company =None):
    if selected_company==None:
        return
    else:
        df = df.drop(['Total','MarketShare'],axis=1)
        df = df[df['Maker']==selected_company]
        df = df.melt(id_vars=['Year', 'Maker'], var_name='Month', value_name='Sales')
        # Concatenate 'Month' and 'Year' columns to create a new column
        df['Month_Year'] = df['Month'].astype(str) + ' ' + df['Year'].astype(str)
        
        # Sort DataFrame by 'Year' and 'Month' columns
        df = df.sort_values(['Year', 'Month'])
        
        # Plot time series bar graph
        fig = px.bar(df, x='Month_Year', y='Sales', title=f'Time Series of Sales for {selected_company} by Month and Year',
                        labels={'Sales': 'Total Sales', 'Month_Year': 'Month and Year'},color='Sales',color_continuous_scale='Viridis')
        
        # Rotate x-axis labels for better readability
        fig.update_layout(xaxis_tickangle=-45,width=1100, height=600)
        
        # Show the plot
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------------
def plot_total_sales_by_maker(df,year='All'):
    if year=='All':
        # Group the DataFrame by 'Maker' and sum the 'Total' sales
        total_sales_by_maker = df.groupby('Maker')['Total'].sum().reset_index()
        
    else:
        total_sales_by_maker = df[df['Year']==year].groupby('Maker')['Total'].sum().reset_index()

    fig = px.bar(total_sales_by_maker, x='Maker', y='Total', color='Maker',
                labels={'Total': 'Total Sales', 'Maker': 'Maker'},
                title='Total Sales by Maker for {} year/s'.format(year))
    fig.update_layout(xaxis={'categoryorder': 'total descending'},width=500, height=400)
    # Group the filtered DataFrame by 'Maker' and sum the 'Total' sales
    total_sales_by_maker = df.groupby('Maker')['Total'].sum().reset_index()

    st.plotly_chart(fig)

def plot_annulus_market_share1(df,year='All'):
    if year=='All':
        # Group the DataFrame by 'Maker' and calculate the market share
        market_share_by_maker = df.groupby('Maker')['Total'].sum() / df['Total'].sum() * 100
        market_share_by_maker = market_share_by_maker.reset_index()
    else:
        market_share_by_maker = df[df['Year']==year].groupby('Maker')['Total'].sum() / df[df['Year']==year]['Total'].sum() * 100
        market_share_by_maker = market_share_by_maker.reset_index()

    # Create the annulus chart (pie chart with an empty central part)
    fig = go.Figure(go.Pie(
        labels=market_share_by_maker['Maker'],
        values=market_share_by_maker['Total'],
        hole=0.4,  # Set the size of the central hole (0 = no hole, 1 = full hole)
        marker=dict(colors=px.colors.qualitative.Pastel,
                    line=dict(color='#000000', width=2))
    ))

    fig.update_layout(
        title="Market Share by Maker for {} year/s".format(year),
        font=dict(size=12),
        width=500, height=500
    )

    st.plotly_chart(fig)

def plot_yearly_sales(df):
    # Group data by Year and sum the sales
    yearly_sales = df.groupby('Year')['Total'].sum().reset_index()

    # Plot year-wise sales
    fig = px.bar(yearly_sales, x='Year', y='Total', title='Yearly Sales',color='Total',labels={'Total':'Total Sales'},color_continuous_scale='Viridis',width=600, height=500)
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Total Sales')
    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.title("Electric Vehicle Sales Dashboard")

    df = pd.read_csv("all_year2.csv")
    df = df.drop(['Unnamed: 0'],axis=1)

    # Sidebar
    st.sidebar.header("Options")
    action = st.sidebar.selectbox("Select Action", ["Scrape EV Sales Data", "Plot Sales Trend"])

    # Select year for filtering
    year = st.selectbox("Select Year", ['All'] + df['Year'].unique().tolist())
    total_sales = st.checkbox("Plot Total Sales Prediction")
    if not total_sales:
        # Get unique company names
        sales_data = df
        company_names = sales_data['Maker'].unique()
        selected_company = st.selectbox("Select a Company", company_names)
    else:
        selected_company = None
    # Create columns layout
    col1, col2, col3 = st.columns([1, 1, 1])


    # Column 1: Total Sales by Maker and Market Share by Maker
    with col1:
        st.header("Total Sales by Maker")
        plot_total_sales_by_maker(df, year)

        # Plot time series of sales for selected company
        st.header("Time Series of Sales")
        plot_sales_time_series(df, selected_company)

    # Column 2: Sales Trend, and Time Series of Sales
    with col2:
        
        # Plot total sales prediction
        
        # Plot sales trend
        st.header("Sales Trend")
        plot_sales_trend2(df, total_sales, selected_company)

        

    with col3:

        st.header("Market Share by Maker")
        plot_annulus_market_share1(df, year)


        st.header("Yearly Sales")
        plot_yearly_sales(df)

if __name__ == "__main__":
    main()
