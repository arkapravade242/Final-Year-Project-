# Polarity Calculation
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt

# Load data from Excel
file_path = "Hastags.xlsx"  # Update with your file path
df = pd.read_excel(file_path)

# Assuming columns: 'Source', 'Target', 'Number' (where 'Number' is the assigned number if available)
G = nx.Graph()

# Add edges from 'Source' and 'Target' columns
for _, row in df.iterrows():
    G.add_edge(row['Source'], row['Target'])

# Initialize labels with given numbers
labels = {}
for _, row in df.iterrows():
    if not pd.isna(row['Number']):  # If number is available
        labels[row['Source']] = int(row['Number'])

# Label Propagation Algorithm
def label_propagation(G, labels, max_iter=100):
    # Initialize remaining nodes with unique labels
    for node in G.nodes():
        if node not in labels:
            labels[node] = -1  # Unassigned nodes get -1 initially

    for _ in range(max_iter):
        nodes = list(G.nodes())
        random.shuffle(nodes)  # Shuffle for randomness

        for node in nodes:
            if labels[node] == -1:  # Only update unassigned nodes
                neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node) if labels[neighbor] != -1]
                if neighbor_labels:
                    labels[node] = max(set(neighbor_labels), key=neighbor_labels.count)

    return labels

# Run label propagation
final_labels = label_propagation(G, labels)

# Print results
print("Final Labels:", final_labels)

# Save results to Excel
result_df = pd.DataFrame(final_labels.items(), columns=['Node', 'Assigned Number'])
result_df.to_excel("output.xlsx", index=False)

# Visualizing the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=list(final_labels.values()), cmap=plt.cm.Set1)
plt.show()
