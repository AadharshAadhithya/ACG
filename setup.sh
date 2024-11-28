#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define helper functions
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed. Please install it first."
        exit 1
    fi
}

# Check for required tools
check_command python
check_command pip
check_command dvc
check_command aws

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found in the root directory. Please create one with AWS credentials."
    exit 1
fi

# Check that AWS credentials are set
if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" || -z "$AWS_DEFAULT_REGION" ]]; then
    echo "Error: AWS credentials are not properly set in .env."
    exit 1
fi

# Pull pre_1 data using DVC
echo "Pulling pre_1 data from DVC..."
dvc pull data/pre_1.dvc

# Success message
echo "Setup complete. pre_1 data is now available!"
