import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_predictions(model, loader):
    """
    Get predictions for all data samples in the given loader using the specified model.

    Args:
        model (torch.nn.Module): The model used for prediction.
        loader (torch.utils.data.DataLoader): The data loader containing the samples.

    Returns:
        tuple: A tuple containing two lists - true_labels and predicted_labels.
            - true_labels (list): The true labels of the data samples.
            - predicted_labels (list): The predicted labels for the data samples.
    """
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            predicted = out.argmax(dim=1)
            true_labels.extend(data.y.tolist())
            predicted_labels.extend(predicted.tolist())
    return true_labels, predicted_labels

    
def get_accuracy(true_labels, predicted_labels):
    """Calculates and returns the accuracy."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
    
def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate various metrics for evaluating the performance of a classification model.

    Args:
        true_labels (array-like): The true labels of the samples.
        predicted_labels (array-like): The predicted labels of the samples.

    Returns:
        dict: A dictionary containing the calculated metrics.
            - accuracy (float): The accuracy of the model.
            - precision (float): The precision of the model.
            - recall (float): The recall of the model.
            - f1 (float): The F1 score of the model.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    return metrics
    
def print_classification_metrics(true_labels, predicted_labels):
    print(classification_report(true_labels, predicted_labels))
    
def plot_confusion_matrix(true_labels, predicted_labels, classes, filename='confusion_matrices/confusion_matrix.png'):
    """
    Plot the confusion matrix based on the true labels and predicted labels.

    Args:
        true_labels (array-like): The true labels.
        predicted_labels (array-like): The predicted labels.
        classes (array-like): The list of class labels.
        filename (str, optional): The filename to save the confusion matrix plot. Defaults to 'confusion_matrices/confusion_matrix.png'.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate confusion matrix plot
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()
    plt.close()