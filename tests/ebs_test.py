import requests
import time
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# URL of the running Flask app on AWS (replace with your Elastic Beanstalk URL)
url = "http://serve-sentiment-env.eba-rpiqddmp.us-east-1.elasticbeanstalk.com/predict/"

# List of inputs: two fake news, two real news
test_inputs = [
    "This is fake news about an event that never happened.",
    "The aliens are coming to kill us.",
    "This is real news about an event that happened.", 
    "Red, blue and yellow are primary colours.",
]

# Number of iterations for each input (100 API calls)
num_iterations = 100

def test_performance():
    """Test the latency of my API"""
    with open('api_latency_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['input', 'start_time', 'end_time', 'latency_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Perform API calls for each input text
        for input in test_inputs:
            for x in range(num_iterations):
                encoded_input = requests.utils.quote(input)
                full_url = url + encoded_input

                start_time = datetime.now()
                response = requests.get(full_url)
                end_time = datetime.now()

                latency = (end_time - start_time).total_seconds()
                
                # Save results to CSV
                writer.writerow({
                    'input': input,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'latency_seconds': latency
                })

def create_boxplot():
    """Create boxplot displaying the latency of my API"""
    # Load the CSV file with latency results
    df = pd.read_csv('api_latency_results.csv')

    # Calculate average latency for each test case
    average_latency = df.groupby('input')['latency_seconds'].mean()
    print("Average Latency:")
    print(average_latency)

    # Generate a boxplot to visualize latency for each test case
    plt.figure(figsize=(10, 6))
    df.boxplot(column='latency_seconds', by='input', grid=False, showfliers=False)

    # Customize the plot
    plt.title('Latency Boxplot')
    plt.xlabel('Test Case')
    plt.ylabel('Latency (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the boxplot as an image
    plt.savefig('latency_boxplot.png')

    # Display the plot
    plt.show()

if __name__ == "__main__":
    #test_performance()
    create_boxplot()