from langchain.utilities import DuckDuckGoSearchAPIWrapper
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd



RESULTS_PER_QUESTION = 4

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

urls= web_search("Identify the industry in which Canoo operates, along with its size, growth rate, trends, and key players ")

#urls=web_search("Gather information on Canoo's financial performance, including its revenue, profit margins, return on investment, and expense structure")

#urls= web_search("Analyze Canoo's main competitors, including their market share, products or services offered, pricing strategies, and marketing efforts.")

#urls=web_search("Gather information on Canoo's financial performance, including its revenue, profit margins, return on investment, and expense structure.")





def scrape_text(urls):
    data = []
    for url in urls:
        try:
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the content of the request with BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract all text from the webpage
                page_text = soup.get_text(separator=" ", strip=True)

                # Store the scraped text and URL in a dictionary
                data.append({"URL": url, "Text": page_text})
            else:
                print(f"Failed to retrieve the webpage: Status code {response.status_code}")
        except Exception as e:
            print(f"Failed to retrieve the webpage: {e}")
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Export the DataFrame to a CSV file
    df.to_csv("scraped_data.csv", index=False)


'''def scrape_text(urls, csv_file="scraped_data.csv"):
    try:
        # Check if the CSV file already exists
        try:
            existing_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            existing_df = pd.DataFrame()

        data = []

        # Scrape text from each URL
        for url in urls:
            try:
                response = requests.get(url)

                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the content of the request with BeautifulSoup
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract all text from the webpage
                    page_text = soup.get_text(separator=" ", strip=True)

                    # Store the scraped text and URL in a dictionary
                    data.append({"URL": url, "Text": page_text})
                else:
                    print(f"Failed to retrieve the webpage: Status code {response.status_code}")
            except Exception as e:
                print(f"Failed to retrieve the webpage: {e}")

        # Convert the list of dictionaries to a DataFrame
        new_df = pd.DataFrame(data)

        # Append the new data to the existing DataFrame
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Export the combined DataFrame to a CSV file
        combined_df.to_csv(csv_file, index=False)
    except Exception as e:
        print(f"Error occurred: {e}")'''

#urls=web_search(query)
scrape_text(urls)

