import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import csv
import os

st.set_option('deprecation.showPyplotGlobalUse', False)
# Function to retrieve URLs for each fiscal year
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

# Function to extract EV 2W sales data from a given URL
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

# Function to scrape EV sales data
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


def plot_sales_trend(csv_file):
    # Load the data
    sales_data = pd.read_csv(csv_file)
    sentiment_data = pd.read_excel("sentiment_scores_combined.xlsx")

    def str_to_num(x):
        if type(x) == str:
            return int(x.replace(",", ""))

    sales_data[['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar','Total']] = sales_data[['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar','Total']].applymap(str_to_num)

    # Calculate average compound score and sum of total sales for each year
    avg_compound_score = sentiment_data.groupby('Year')['Compound Score'].mean()
    sum_total_sales = sales_data.groupby('Year')['Total'].sum()

    # Convert to DataFrame
    data = pd.DataFrame({'Avg Compound Score': avg_compound_score, 'Total Sales': sum_total_sales}).reset_index()

    # Extract numeric years
    data['Year'] = data['Year'].str.extract(r'(\d+)').astype(int)

    # Prepare data for polynomial regression
    X = data['Year'].values.reshape(-1, 1)  # Year
    y = data['Total Sales'].values.reshape(-1, 1)  # Sales

    # Fit polynomial regression model
    degree = 2  # or choose an appropriate degree
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predictions for upcoming years
    future_years = np.arange(data['Year'].min(), data['Year'].max()+2).reshape(-1, 1)
    future_years_poly = poly_features.transform(future_years)
    sales_pred = model.predict(future_years_poly)

    upcoming_years = np.array([[data['Year'].max()+1], [data['Year'].max()+2]])
    #upcoming_years = np.array([[2024], [2025]])
    future_years_poly = poly_features.transform(upcoming_years)
    sales_predf = model.predict(future_years_poly)

    predicted_sales = pd.DataFrame({'Year': upcoming_years.flatten(), 'Predicted Sales': sales_predf.flatten()})

    # Calculate upper and lower bounds of the error margins
    sales_pred_upper = sales_predf + np.std(y)
    sales_pred_lower = sales_predf - np.std(y)
    # Plot the data, trend line, and error margins
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Sales')

    plt.plot(upcoming_years, sales_predf, color='blue', label='Trend Line')
    plt.plot(future_years, sales_pred, color='red', label='Trend Line')
    plt.errorbar(predicted_sales['Year'], predicted_sales['Predicted Sales'], yerr=np.std(y), fmt='o', color='blue', label='Predicted Sales')

    plt.fill_between(upcoming_years.flatten(), sales_pred_upper.flatten(), sales_pred_lower.flatten(), color='gray', alpha=0.3, label='Error Margins')
    plt.xlabel('Year')
    plt.ylabel('Total Sales')
    plt.title('Trend of Total Sales over Years with Error Margins')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    st.pyplot()

# Main Streamlit app
def main():
    st.title("Electric Vehicle Sales Analysis")

    # Sidebar
    st.sidebar.header("Options")
    action = st.sidebar.selectbox("Select Action", ["Scrape EV Sales Data", "Plot Sales Trend"])

    if action == "Scrape EV Sales Data":
        st.subheader("Scrape EV Sales Data")
        csv_file = st.text_input("Enter CSV file path", "all_year.csv")
        if st.button("Scrape Data"):
            scrape_ev_sales(csv_file)

    elif action == "Plot Sales Trend":
        st.subheader("Plot Sales Trend")
        csv_file = st.text_input("Enter CSV file path", "all_year.csv")
        if st.button("Plot Trend"):
            plot_sales_trend(csv_file)

if __name__ == "__main__":
    main()
