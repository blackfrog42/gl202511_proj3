# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory to register')
    parser.add_argument("--model_info_output_path", type=str, required=True, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering", args.model_name)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    loaded_model = mlflow.sklearn.load_model(model_uri=str(model_path))

    model_info = mlflow.sklearn.log_model(
        sk_model=loaded_model,
        artifact_path="model",
    )

    model_uri = model_info.model_uri
    model_version = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    client = MlflowClient()
    while model_version.status == "PENDING_REGISTRATION":
        time.sleep(5)
        model_version = client.get_model_version(name=args.model_name, version=model_version.version)

    if model_version.status != "READY":
        raise RuntimeError(f"Model registration failed with status: {model_version.status}")

    registered_version = int(model_version.version)

    output_path = Path(args.model_info_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump({
            "model_name": args.model_name,
            "model_version": int(registered_version),
        }, f)


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