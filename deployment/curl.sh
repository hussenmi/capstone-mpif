#!/bin/bash

# Replace with the appropriate URL of your Flask app
URL="http://localhost:5000/"

# Replace with the paths to the CSV files you want to upload
COORDINATES_PATH="test_files/SP41_2_X2Y8/BaselTMA_SP41_2_X2Y8_coords.csv"
EXPRESSIONS_PATH="test_files/SP41_2_X2Y8/BaselTMA_SP41_2_X2Y8_expression.csv"

# Perform the HTTP POST request
curl -X POST -F "coordinates=@$COORDINATES_PATH" -F "expressions=@$EXPRESSIONS_PATH" $URL