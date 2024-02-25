from create_graphs_for_classification import CellGraphDataset
import torch
from torch_geometric.loader import DataLoader

def load_cell_data(data_dir, batch_size=32):
    """
    Load cell data from the specified directory.

    Args:
        data_dir (str): The directory path where the cell data is located.
        batch_size (int, optional): The batch size for the data loader. Defaults to 32.

    Returns:
        tuple: A tuple containing the dataset, train loader, and test loader.
            - dataset (CellGraphDataset): The loaded cell graph dataset.
            - train_loader (DataLoader): The data loader for the training set.
            - test_loader (DataLoader): The data loader for the test set.
    """
    dataset = CellGraphDataset(root=data_dir, response_label_dict=None)

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    # Define the size of your training set
    train_size = int(len(dataset) * 0.8)  # e.g., 80% of the data
    test_size = len(dataset) - train_size  # The rest for testing

    # Split the dataset
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, test_loader