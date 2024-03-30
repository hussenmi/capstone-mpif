# Enhancing Immunotherapy Predictions with Graph Neural Networks

## Project Overview

A video explanation of the project can be found [here](https://screenapp.io/app/#/shared/2ca54bdd-b604-4bb0-bbdc-248efe1eabd5).

This repository contains the work for a research project focused on personalizing medicine in breast cancer treatment through the predictive analysis of tumor receptors using Graph Neural Networks (GNNs). By accurately identifying receptors like ER, PR, and HER2 within tumor cells, this project aims to contribute to the development of personalized therapeutic approaches. The work contains the analysis of tumor microenvironments from multiplex immunofluorescence imaging (mpIF) data using different machine learning methods including different forms of graph neural networks.

## Models Used
- **SPACE-GM**: A model proposed in [this paper](https://www.nature.com/articles/s41568-023-00582-6) that mainly uses a Graph Isomorphism Network (GIN) as its backbone
straightforward convolutional neural network architecture designed as a baseline for the image classification task.
- **Proposed Method**: The other method tested uses a Graph Convolutional Network for the task of graph classification. Its implementation is found in `training/model.py`.

## Experimentation and Model Selection
Throughout the training process, model parameters, different metrics, and artifacts were logged to MLFlow. This allows for the comparison of models later on. Key metrics such as accuracy, precision, recall, and F1 score were monitored. The confusion matrices for these runs were also logged. The best-performing model for each of the three receptors, as determined by these metrics, was saved for future use.

## Project Structure Overview
The project has two main folders -- `training` and `deployment`. The `training` directory houses all necessary scripts and files for training the models. Due to size constraints, the dataset is not included in the repository. This folder also includes scripts for evaluating different models and hyperparameter combinations, with functionality to save the best-performing model based on predefined criteria for future use. The `deployment` directory contains all the necessary files required to deploy the model on a Flask server.


## How to Use the Deployment

In order to get the deployment up and running, we use the `deployment` directory. To bring up the server, from the root directory, use the following commands:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python -m deployment.app
```

To test the application, you can use two different approaches.

### 1. Web App Interface
Since the model weights are provided in this repository, users can interact with a web-based interface to upload CSV files for the coordinates and expressions of the regions and receive predictions directly in the browser. The coordinates file needs to have columns `X` and `Y`. The expression file needs the columns `ACQUISITION_ID`, `CELL_ID`, and the biomarkers as separate columns. 

Navigate to the provided URL (e.g., `http://localhost:5001`) and use the upload form to select the files. Example files are found in `deployment/test_files`. Upon submission, the predictions will be displayed.

### 2. API Endpoint
For programmatic access, users can send the coordinates and expressions files via a `curl` command to an API endpoint and receive predictions in a JSON response format. Here is the structure of the response:

```json
{
  "erStatus": {
    "prediction": "<Present or Absent>",
    "probability": "<Probability>"
  },
  "prStatus": {
    "prediction": "<Present or Absent>",
    "probability": "<Probability>"
  },
  "her2Status": {
    "prediction": "<Present or Absent>",
    "probability": "<Probability>"
  }
}
```


To use this method, one can run these commands:

```bash
cd deployment
./curl.sh
```

Make sure to run `chmod +x curl.sh` to make the file executable.