#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Load environment variables from .env file
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

echo "Creating Conda environment: omnianswer"
conda env create -f omnianswer-conda-env.yaml -y

echo "Installing Guardrails Hub modules"
# Use 'conda run' to execute commands within the new environment
conda run -n omnianswer guardrails hub install hub://guardrails/toxic_language hub://guardrails/ban_list

echo "Setup complete!"