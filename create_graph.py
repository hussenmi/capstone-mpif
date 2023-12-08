import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import distance_matrix
import networkx as nx

# Read in the data
coordinates_with_id = pd.read_csv("/juno/work/shah/users/ibrahih3/codebase/space-gm/data_melanoma/df_0_1_1_coords.csv")
expressions_with_id = pd.read_csv("/juno/work/shah/users/ibrahih3/codebase/space-gm/data_melanoma/df_0_1_1_expression.csv")

coordinates = coordinates_with_id[['X', 'Y']]
# biomarkers = expressions_with_id.columns[2:]


def create_graph_from_coordinates(coordinates_df, threshold, expressions_df):
    """
    Create a graph from the coordinates of the cells.
    
    Args:
        coordinates_df (pd.DataFrame): A DataFrame with the columns 'X' and 'Y'.
        threshold (int): The distance threshold for two cells to be connected.
        expressions_df (pd.DataFrame): A DataFrame containing the columns ACQUISITION_ID, CELL_ID, and the biomarkers.
        
    Returns:
        nx.Graph: A graph with the cells as nodes and edges between cells that are within the distance threshold.
    """
    # Compute the distance matrix
    dist_matrix = distance_matrix(coordinates_df.values, coordinates_df.values)
    
    # Create an adjacency matrix based on the distance threshold
    adjacency_matrix = (dist_matrix < threshold).astype(int)
    
    # Convert adjacency matrix to DataFrame
    adjacency_df = pd.DataFrame(adjacency_matrix, index=coordinates_df.index, columns=coordinates_df.index)
    
    # Create a graph from the adjacency matrix
    graph = nx.from_pandas_adjacency(adjacency_df)
    
    # Get the biomarker names
    biomarker_names = expressions_with_id.columns[2:]
    
    # Add node features for each cell
    for node in graph.nodes:
        graph.nodes[node]['features'] = expressions_df.loc[node, biomarker_names].values
    
    return graph


graph = create_graph_from_coordinates(coordinates, 30, expressions_with_id)
    
# save the graph
import pickle

with open('graph_0_1_1.pickle', 'wb') as handle:
    pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
# simple visualization
import matplotlib.pyplot as plt

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(graph, node_size=50, node_color='blue')
plt.show()

# from the graph, see the features of the first node
len(graph.nodes[1]['features'])





