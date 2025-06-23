# Filter Tags
import pandas as pd
import re

# Load the CSV file
file_path = r'C:\Users\hp\OneDrive\Desktop\Graph\Tweets_main dataset\march_1.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Function to extract tags (mentions) from a tweet
def extract_tags(tweet):
    # Check if the tweet is a string, if not return an empty list
    if isinstance(tweet, str):
        # Using regex to find all @mentions in the tweet
        return re.findall(r'@(\w+)', tweet)
    return []  # Return an empty list if the value is not a string

# Assuming the 'tweet' column contains the tweets and 'username' column contains the usernames
df['tags'] = df['tweet'].apply(extract_tags)

# Select only the necessary columns: 'username' and 'tags'
df_filtered = df[['username', 'tags']]

# Save the filtered data to an Excel file
output_file = 'tweet_tags_filtered_march_1_output.xlsx'
df_filtered.to_excel(output_file, index=False)

print(f"Filtered data has been saved to {output_file}")
