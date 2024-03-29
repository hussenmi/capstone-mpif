import os
import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import distance_matrix
import networkx as nx
import torch_geometric.utils as utils

PATH = '../../../data_bodenmiller/data_bodenmiller'


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
    
    biomarker_names = expressions_df.columns[2:]
    
    # Add node features for each cell (node)
    for node in graph.nodes:
        # convert the features to a list
        graph.nodes[node]['features'] = list(expressions_df.loc[node, biomarker_names].values)
    
    return graph


def get_response_dict(response):
    """
    Get a dictionary mapping region IDs to response values.

    Args:
        response (str): The name of the response column.

    Returns:
        dict: A dictionary mapping region IDs to response values.
    """
    response_label = pd.read_csv(f'{PATH}_label/basel_label.csv')
    response_label_dict = {}

    for i in range(len(response_label)):
        response_label_dict[response_label['REGION_ID'][i]] = response_label[response][i]
        
    return response_label_dict

    
class CellGraphDataset(InMemoryDataset):
    """Dataset class for creating cell graph data for classification.

    Args:
        root (str): Root directory where the dataset should be saved.
        response_label_dict (dict, optional): Dictionary mapping region IDs to response labels. Defaults to None.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
        pre_transform (callable, optional): A function/transform that takes in raw data and returns a pre-processed version. Defaults to None.
    """
    def __init__(self, root, response_label_dict=None, transform=None, pre_transform=None):
        self.response_label_dict = response_label_dict if response_label_dict is not None else {}
        super(CellGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No need to download anything
        pass
    
    def process(self):
        data_list = []
        
        for i, region in enumerate(region_ids):
            # Load data for this region
            coordinates_with_id = pd.read_csv(f"{PATH}/basel_data/{region}_coords.csv")
            expressions_with_id = pd.read_csv(f"{PATH}/basel_data/{region}_expression.csv")

            # Create the graph from the data
            graph = create_graph_from_coordinates(coordinates_with_id[['X', 'Y']], 30, expressions_with_id)

            # Convert the graph to a PyTorch geometric data object
            G = utils.from_networkx(graph)

            # Add the features as 'x'
            G.x = torch.tensor([data for _, data in graph.nodes(data='features')], dtype=torch.float)

            # Only add the 'y' value if there are labels
            if self.response_label_dict:
                G.y = torch.tensor([self.response_label_dict[region]])  # Add the graph-level label

            # Save the data object to the list
            data_list.append(G)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == '__main__':
    # Create the datasets for three different labels
    response_label_dict_er = get_response_dict('ERStatus')
    response_label_dict_pr = get_response_dict('PRStatus')
    response_label_dict_her2 = get_response_dict('HER2Status')

    region_ids = list(response_label_dict_er.keys())

    # dataset = CellGraphDataset(root='.')
    dataset_er = CellGraphDataset(root='./ER_status', response_label_dict=response_label_dict_er)
    dataset_pr = CellGraphDataset(root='./PR_status', response_label_dict=response_label_dict_pr)
    dataset_her2 = CellGraphDataset(root='./HER2_status', response_label_dict=response_label_dict_her2)