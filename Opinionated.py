# Import packages
%matplotlib inline
import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
from igraph import *

import numpy as np

def IC_with_polarity(g, S, polarity_map, p=0.4, mc=1000):
    """
    Input:
        g  - Graph object (assumed directed)
        S  - List of seed nodes
        polarity_map - Dictionary mapping nodes to their polarity (e.g., {1: 'A', 2: 'B', 3: 'A', ...})
        p  - Maximum propagation probability (default: 0.5)
        mc - Number of Monte-Carlo simulations (default: 1000)

    Output:
        Expected number of influenced nodes (float)
    """

    spread = []
    rng = np.random.default_rng()  # Use new random generator

    for _ in range(mc):
        A = set(S)  # Set of activated nodes
        new_active = set(S)  # Newly activated nodes
        
        while new_active:
            new_ones = set()
            for node in new_active:
                neighbors = set(g.neighbors(node, mode="out"))
                if not neighbors:
                    continue
                # Assign probability based on polarity (fix)
                probabilities = [
                    1.0 if polarity_map.get((node, neighbor), 0) == 1  # Use tuple key
                    else np.random.uniform(0, p)
                    for neighbor in neighbors
                ]

                # Assign probability based on polarity
                #probabilities = [
                #    1.0 if polarity_map[node] == polarity_map[neighbor] 
                #    else rng.uniform(0, p)
                #   for neighbor in neighbors
                #]
                
                # Determine which neighbors become activated
                success = rng.uniform(0, 1, len(neighbors)) < probabilities
                new_ones.update(np.extract(success, list(neighbors)))

            new_active = new_ones - A  # Only consider newly activated nodes
            A.update(new_active)

        spread.append(len(A))

    return np.mean(spread)


def greedy_with_polarity(g, k, polarity_map, p=0.4, mc=1000):
    """
    Greedy Algorithm with Polarity-Based IC
    """
    S, spread, timelapse, start_time = [], [], [], time.time()
    
    for _ in range(k):
        best_spread = 0
        for j in set(range(g.vcount())) - set(S):
            s = IC_with_polarity(g, S + [j], polarity_map, p, mc)
            if s > best_spread:
                best_spread, node = s, j
        
        S.append(node)
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
    return(S,spread,timelapse)



import time
import numpy as np

def celf_with_polarity(g, k, polarity_map, p=0.4, mc=1000):  
    """
    Implements the CELF (Cost-Effective Lazy Forward) algorithm for influence maximization.
    
    Input:
        g  - Graph object (assumed directed)
        k  - Number of seed nodes
        polarity_map - Dictionary mapping nodes to their polarity
        p  - Maximum propagation probability (default: 0.1)
        mc - Number of Monte-Carlo simulations (default: 1000)

    Output:
        Optimal seed set, resulting spread, time for each iteration, number of lookups
    """
    
    start_time = time.time()

    # --------------------
    # Step 1: Find the first seed node using greedy evaluation
    # --------------------
    marg_gain = [IC_with_polarity(g, [node], polarity_map, p, mc) for node in range(g.vcount())]
    
    # Sort nodes by their influence spread
    Q = sorted(zip(range(g.vcount()), marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and initialize tracking variables
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time() - start_time]
    
    # --------------------
    # Step 2: Find the next k-1 nodes using lazy evaluations
    # --------------------
    for _ in range(k - 1):
        check, node_lookup = False, 0
        
        while not check:
            node_lookup += 1
            current = Q[0][0]

            # Recalculate spread of the top node
            Q[0] = (current, IC_with_polarity(g, S + [current], polarity_map, p, mc) - spread)

            # Sort the list again
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if the top node remains unchanged
            check = (Q[0][0] == current)

        # Select the next seed node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from consideration
        Q = Q[1:]

    return(S, SPREAD, timelapse, LOOKUPS)



import pandas as pd
import igraph as ig

# Load the data
file_path = "Edges_all2.csv"  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Drop missing vas and convert Source to int
data_cleaned = data.dropna().astype(int)

# Extract source and target as lists
Source = data_cleaned["Source"].tolist()
Target = data_cleaned["Target"].tolist()

# Create a unique mapping for node IDs to sequential integer indices
unique_nodes = sorted(set(Source) | set(Target))  # Get all unique node IDs
node_map = {node: idx for idx, node in enumerate(unique_nodes)}  # Map node to index

# Convert Source and Target using this mapping
mapped_source = [node_map[node] for node in Source]
mapped_target = [node_map[node] for node in Target]

# Create the graph
g = ig.Graph(directed=True)
g.add_vertices(len(unique_nodes))  # Add vertices (total unique nodes)
g.add_edges(zip(mapped_source, mapped_target))  # Add edges with mapped indices

# Assign labels (only for small graphs)
g.vs["label"] = unique_nodes if len(unique_nodes) < 500 else None
g.vs["color"] = "#FBB4AE"
g.es["color"] = "#B3CDE3"

# Plot the graph
layout = g.layout("kk")  # Kamada-Kawai layout
ig.plot(g, bbox=(800, 800), margin=20, layout=layout)




# Extract source, target, and polarity from cleaned DataFrame
Source = data_cleaned["Source"].tolist()
Target = data_cleaned["Target"].tolist()
Polarity = data_cleaned["Polarity"].fillna(0).tolist()  # Replace NaN polarity with 0

# Create a mapping for node IDs
unique_nodes = sorted(set(Source) | set(Target))
node_map = {node: idx for idx, node in enumerate(unique_nodes)}

# Convert Source and Target to mapped indices
mapped_source = [node_map[node] for node in Source]
mapped_target = [node_map[node] for node in Target]

# Create polarity_map with (source, target) as keys
polarity_map = {
    (node_map[src], node_map[tgt]): pol
    for src, tgt, pol in zip(Source, Target, Polarity)
}




# Run algorithms with polarity-based IC
celf_output   = celf_with_polarity(g, 2, polarity_map, p=0.4, mc=100)
greedy_output = greedy_with_polarity(g, 2, polarity_map, p=0.4, mc=100)

# Print results
print("celf output:   " + str(celf_output[0]))
print("greedy output: " + str(greedy_output[0]))



# Plot settings
plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

# Plot Computation Time
plt.plot(range(1,len(greedy_output[2])+1),greedy_output[2],label="Greedy",color="#FBB4AE")
plt.plot(range(1,len(celf_output[2])+1),celf_output[2],label="CELF",color="#B3CDE3")
plt.ylabel('Computation Time (Seconds)'); plt.xlabel('Size of Seed Set')
plt.title('Computation Time'); plt.legend(loc=2);




# Plot Expected Spread by Seed Set Size
plt.plot(range(1,len(greedy_output[1])+1),greedy_output[1],label="Greedy",color="#FBB4AE")
plt.plot(range(1,len(celf_output[1])+1),celf_output[1],label="CELF",color="#B3CDE3")
plt.xlabel('Size of Seed Set'); plt.ylabel('Expected Spread')
plt.title('Expected Spread'); plt.legend(loc=2);
