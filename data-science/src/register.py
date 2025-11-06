# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
import json
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
        registered_model_name=args.model_name,
    )

    registered_version = model_info.registered_model_version
    if registered_version is None:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(name=args.model_name)
        if not latest_versions:
            raise RuntimeError(f"Model {args.model_name} failed to register.")
        registered_version = max(int(version.version) for version in latest_versions)

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