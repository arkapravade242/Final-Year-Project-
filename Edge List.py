# Edge List
import pandas as pd

# Load the provided Excel file to inspect its contents
file_path = r'C:\Users\hp\OneDrive\Desktop\Graph\All_Tweets_tags_filtered_output.xlsx'
data = pd.read_excel(file_path)

# Filter rows where tags are not empty
filtered_data = data[data['tags'].apply(lambda x: len(eval(x)) > 0)]

# Prepare the new DataFrame where each connection (username, tag) will be in a separate row
expanded_data = []

# For each username and its associated tags, create a row for each connection
for index, row in filtered_data.iterrows():
    username = row['username']
    tags = eval(row['tags'])  # Convert the string representation of list back to a list
    for tag in tags:
        expanded_data.append({'Source': username, 'Target': tag})

# Convert to DataFrame
connections_df = pd.DataFrame(expanded_data)

# Save to a new Excel file
output_file = r'C:\Users\hp\OneDrive\Desktop\Graph\Tweets_filtered/username_tag_connections_march_1.xlsx'
connections_df.to_excel(output_file, index=False)

# Return the path to the newly created file
output_file
