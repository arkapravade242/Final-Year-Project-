# Hashtgs Extract
import pandas as pd
import re

# Load the CSV file
file_path = r'C:\Users\hp\OneDrive\Desktop\Graph\Tweets_main dataset\february_1.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Function to extract hashtags from the 'tweet' column
def extract_hashtags(text):
    """Extract hashtags from a given text."""
    if isinstance(text, str):  # Ensure text is a string before applying regex
        return re.findall(r"#\w+", text)
    return []  # Return empty list if not a string

# Apply the function to the 'tweet' column to extract hashtags
df['extracted_hashtags'] = df['tweet'].apply(extract_hashtags)

# Flatten the list of all hashtags, keeping the original order
ordered_hashtags = [hashtag for sublist in df['extracted_hashtags'] for hashtag in sublist]

# Save the ordered hashtags into a DataFrame and export it to an Excel file
hashtags_df = pd.DataFrame(ordered_hashtags, columns=["Hashtags"])

# Export to an Excel file
output_path = 'ordered_hashtags_from_tweets_february1.xlsx'
hashtags_df.to_excel(output_path, index=False)

print(f"Ordered hashtags have been saved to {output_path}")
