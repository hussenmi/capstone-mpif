import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifact, log_artifacts
from model import GCN
from train import train_network
from data_loader import load_cell_data
from utils import calculate_metrics, print_classification_metrics, get_all_predictions, plot_confusion_matrix
import torch
import random


def run_experiment(receptor_type, model_name, hyperparams, force_train=False):
    """
    Run an experiment with the specified parameters.

    Args:
        receptor_type (str): The type of receptor.
        model_name (str): The name of the model.
        hyperparams (dict): A dictionary containing hyperparameters for the experiment.
        force_train (bool, optional): Whether to force training even if a pre-trained model exists. Defaults to False.
    """
    epochs, lr, momentum, batch_size, hidden_channels, optimizer_choice = hyperparams.values()
    # print(f"Running experiment with: model_name={model_name}, epochs={epochs}, lr={lr}, momentum={momentum}, batch_size={batch_size}")
    with mlflow.start_run():
        log_params({
            "model_name": model_name,
            "epochs": epochs,
            "lr": lr,
            "momentum": momentum,
            "batch_size": batch_size,
            "hidden_channels": hidden_channels,
            "optimizer_choice": optimizer_choice
        })
        
        # Load the dataset and create data loaders       
        dataset, trainloader, testloader = load_cell_data(data_dir=f'./{receptor_type}', batch_size=batch_size)
        
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes

        if model_name == "GCN":
            print('num_node_features:', num_node_features)
            print('num_classes:', num_classes)
            net = GCN(num_node_features, num_classes, hidden_channels)
        # elif model_name == "CNNModel2":
        #     net = CNNModel2()
         
        # Training
        model_filename = f"model_weights/{model_name}_receptor{receptor_type}_lr{lr}_momentum{momentum}_epochs{epochs}_batchsize{batch_size}_hiddenchannels{hidden_channels}_optimizer{optimizer_choice}.pth"
        train_network(net, trainloader, model_path=model_filename, epochs=epochs, lr=lr, momentum=momentum, optimizer_choice=optimizer_choice, force_train=force_train)
        net.load_state_dict(torch.load(model_filename))  # Ensure using the trained model

        
        true_labels, predicted_labels = get_all_predictions(net, testloader)
        print(true_labels, predicted_labels)
        print_classification_metrics(true_labels, predicted_labels)
        metrics = calculate_metrics(true_labels, predicted_labels)

        # Log metrics
        for metric_name, value in metrics.items():
            log_metric(metric_name, value)
        
        # Save and log confusion matrix plot with a dynamic filename
        plot_filename = f"confusion_matrices/{model_name}_confusion_matrix_receptor{receptor_type}_lr{lr}_momentum{momentum}_epochs{epochs}_batchsize{batch_size}_hiddenchannels{hidden_channels}_optimizer{optimizer_choice}.png"
        classes = ['Negative', 'Positive']  # Replace with your actual class names
        plot_confusion_matrix(true_labels, predicted_labels, classes, filename=plot_filename)

        # Log the confusion matrix image as an MLflow artifact
        log_artifact(plot_filename)




if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("MPIF_Experiments")
    
    # Define the hyperparameter search space
    
    receptors = ['ER_status', 'PR_status', 'HER2_status']
    
    for receptor_type in receptors:
        print(f"Running experiments for receptor type: {receptor_type}")
        
        # Define the hyperparameter search space
        # hyperparameters_simplecnn = {
        #     "epochs": [5, 7, 10, 15],
        #     "lr": [0.001, 0.0005, 0.0001],
        #     "momentum": [0.9, 0.95, 0.99, 0.999],
        #     "batch_size": [8, 16, 32, 64, 128],
        #     "hidden_channels": [16, 32, 64],
        #     "optimizer": ["SGD", "Adam"]
        # }

        hyperparameters_gcn = {
            "epochs": [60, 80],
            "lr": [0.001, 0.01, 0.005],
            "momentum": [0.9, 0.95, 0.99, 0.999],
            "batch_size": [16, 32, 64, 128],
            "hidden_channels": [64],
            "optimizer": ["Adam"]
        }
        
        # models_to_test = [('SimpleCNN', hyperparameters_simplecnn), ('CNNModel2', hyperparameters_cnnmodel2)]
        models_to_test = [('GCN', hyperparameters_gcn)]
        
        experiments_per_model = 3

        for model_name, hyperparams_space in models_to_test:
            # print(model_name, hyperparams_space)
            for _ in range(experiments_per_model):
                # Sample a random set of hyperparameters for each experiment
                hyperparams = {k: random.choice(v) for k, v in hyperparams_space.items()}
                
                print(f"Running experiment with: {model_name}, hyperparams: {hyperparams}")
                
                run_experiment(receptor_type, model_name, hyperparams, force_train=False)