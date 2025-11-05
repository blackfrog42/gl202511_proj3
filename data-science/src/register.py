# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)


    # -----------  WRITE YOUR CODE HERE -----------
    
    # Step 1: Load the model from the specified path using `mlflow.sklearn.load_model` for further processing. 
    #model = mlflow.sklearn.load_model(args.model_path)
    resolved_model_path = args.model_path.replace("${{name}}", os.getenv("AZUREML_RUN_NAME", ""))
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Resolved model path not found: {resolved_model_path}")
    model = mlflow.sklearn.load_model(resolved_model_path)

    # Step 2: Log the loaded model in MLflow with the specified model name for versioning and tracking.  
    mlflow.sklearn.log_model(model, args.model_name)

    # Step 3: Register the logged model using its URI and model name, and retrieve its registered version.  
    model_uri = f"models:/{args.model_name}/latest"
    mlflow.register_model(model_uri, args.model_name)

    # Step 4: Write model registration details, including model name and version, into a JSON file in the specified output path.  
    model_info = {
        "model_name": args.model_name,
        "model_version": mlflow.registered_model.get_latest_version(args.model_name)
    }

    with open(args.model_info_output_path, 'w') as f:
        json.dump(model_info, f)


if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()