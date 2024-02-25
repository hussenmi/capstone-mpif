#!/bin/bash

URL="http://localhost:5000/"

COORDINATES_PATH="test_files/SP41_129_X7Y8/BaselTMA_SP41_129_X7Y8_coords.csv"
EXPRESSIONS_PATH="test_files/SP41_129_X7Y8/BaselTMA_SP41_129_X7Y8_expression.csv"

curl -X POST -F "coordinates=@$COORDINATES_PATH" -F "expressions=@$EXPRESSIONS_PATH" $URL