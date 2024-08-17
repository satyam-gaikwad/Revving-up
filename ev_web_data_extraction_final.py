# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:59:40 2024

@author: shaha
"""

import requests
from bs4 import BeautifulSoup
import csv

# Define headers with a user-agent to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Function to extract EV 2W sales data from a given URL
def extract_ev_2w_sales(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, headers=headers)
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
            print("No table found on the webpage.")
            return None

    except Exception as e:
        print(f"An error occurred while extracting data from {url}: {e}")
        return None

# Column names
columns = ['Year', 'Maker', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'MarketShare']

# List of URLs for each fiscal year
urls = [
    "https://www.smev.in/fy-20",
    "https://www.smev.in/fy-21",
    "https://www.smev.in/fy-21-22",
    "https://www.smev.in/fy-22-23",
    "https://www.smev.in/fy-23-24"
]

# Open the CSV file in write mode
with open("all_year.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Write column names to the CSV file
    writer.writerow(columns)

    # Iterate over each URL and extract data
    for url in urls:
        print(f"Processing URL: {url}")
        data = extract_ev_2w_sales(url)

        if data:
            # Extract year from the URL
            year = url.split('/')[-1]

            # Write the extracted data into the CSV file
            for row in data[1:]:  # Skip the first row (headers)
                writer.writerow([year] + row)

            print(f"Data for {year} extracted and added to the CSV file")
        else:
            print(f"No data extracted from {url}")

print("Process completed.")

