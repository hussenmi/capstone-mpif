import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_predictions(model, loader):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            predicted = out.argmax(dim=1)
            true_labels.extend(data.y.tolist())  # No need to convert labels to class indices
            predicted_labels.extend(predicted.tolist())
    return true_labels, predicted_labels

# for data in test_loader:  # Iterate in batches over the test dataset.
#     data = data.to(device)

#     out = model(data.x, data.edge_index, data.batch)  
#     pred = out.argmax(dim=1)  # Use the class with highest probability.
#     y_true.extend(data.y.tolist())  # Collecting true labels
#     y_pred.extend(pred.tolist())  # Collecting predicted labels
    
def get_accuracy(true_labels, predicted_labels):
    """Calculates and returns the accuracy."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
    
def calculate_metrics(true_labels, predicted_labels):
    
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