from flask import Flask, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from training.create_graphs_for_classification import create_graph_from_coordinates
from training.model import GCN
from torch_geometric.data import Data
import torch
from torch.nn.functional import softmax

app = Flask(__name__)

THRESHOLD = 30
NUM_NODE_FEATURES = 30
NUM_CLASSES = 2
PATH_ER_MODEL = 'training/model_weights/GCN_receptorER_status_lr0.001_momentum0.999_epochs60_batchsize64_hiddenchannels64_optimizerAdam.pth'
PATH_PR_MODEL = 'training/model_weights/GCN_receptorPR_status_lr0.01_momentum0.9_epochs60_batchsize128_hiddenchannels64_optimizerAdam.pth'
PATH_HER2_MODEL = 'training/model_weights/GCN_receptorHER2_status_lr0.001_momentum0.95_epochs60_batchsize64_hiddenchannels64_optimizerAdam.pth'

def load_model(model_path):
    # Load the best trained model
    model = GCN(NUM_NODE_FEATURES, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'coordinates' not in request.files or 'expressions' not in request.files:
            return jsonify({'error': 'Missing files'}), 400
        
        # The form has been submitted, process the files and make predictions
        coordinates_file = request.files['coordinates']
        expressions_file = request.files['expressions']

        # Read the uploaded files into pandas DataFrames
        coordinates_df = pd.read_csv(coordinates_file)
        coordinates_df = coordinates_df[['X', 'Y']]
        expressions_df = pd.read_csv(expressions_file)

        # Create a graph from the data
        graph = create_graph_from_coordinates(coordinates_df, THRESHOLD, expressions_df)
        
        # Convert the graph to a PyTorch Geometric Data object
        G = Data(edge_index=torch.tensor(list(graph.edges)).t().contiguous())

        # Add the features as 'x'
        G.x = torch.tensor([data for _, data in graph.nodes(data='features')], dtype=torch.float)
        
        # Create a 'batch' tensor
        G.batch = torch.zeros(G.x.size(0), dtype=torch.long)

        # Load the trained models
        model_er = load_model(PATH_ER_MODEL)
        model_pr = load_model(PATH_PR_MODEL)
        model_her2 = load_model(PATH_HER2_MODEL)

        # Make predictions
        with torch.no_grad():
            logits_er = model_er(G.x, G.edge_index, G.batch)
            logits_pr = model_pr(G.x, G.edge_index, G.batch)
            logits_her2 = model_her2(G.x, G.edge_index, G.batch)
            
            # Convert outputs to probabilities
            probabilities_er = torch.nn.functional.softmax(logits_er, dim=1)
            probabilities_pr = torch.nn.functional.softmax(logits_pr, dim=1)
            probabilities_her2 = torch.nn.functional.softmax(logits_her2, dim=1)

            # Choose the class with the highest probability
            prediction_er = probabilities_er.argmax().item()
            prediction_pr = probabilities_pr.argmax().item()
            prediction_her2 = probabilities_her2.argmax().item()
            
            # Get probabilities for the predicted classes
            er_probability = probabilities_er.max().item() * 100
            pr_probability = probabilities_pr.max().item() * 100
            her2_probability = probabilities_her2.max().item() * 100
            
            er_status = "Present" if prediction_er == 1 else "Absent"
            pr_status = "Present" if prediction_pr == 1 else "Absent"
            her2_status = "Present" if prediction_her2 == 1 else "Absent"

            
        # Check if request comes from a command-line tool like curl
        if 'curl' in request.headers.get('User-Agent', ''):
            # Return JSON response for curl-like requests
            return jsonify({
                'erStatus': {
                    'prediction': er_status,  # "Present" or "Absent"
                    'probability': f"{probabilities_er.max().item():.3f}"
                },
                'prStatus': {
                    'prediction': pr_status,
                    'probability': f"{probabilities_pr.max().item():.3f}"
                },
                'her2Status': {
                    'prediction': her2_status,
                    'probability': f"{probabilities_her2.max().item():.3f}"
                }
            })
        else:
            return render_template('results.html', 
                       er_status=f"{er_status} ({er_probability:.1f}%)", 
                       pr_status=f"{pr_status} ({pr_probability:.1f}%)", 
                       her2_status=f"{her2_status} ({her2_probability:.1f}%)")

    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)