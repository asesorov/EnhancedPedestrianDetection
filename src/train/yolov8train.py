import os
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')

root_dir = Path(__file__).resolve().parents[1]  # root directory absolute path
data_dir = os.getenv('DATASET_DIR')
data_yaml_path = os.path.join(data_dir, "data.yaml")
metrics_path = os.path.join(root_dir, 'reports/train_metrics.json')


def rename_dict_keys(dictionary):
    renamed_dict = {}
    for key, value in dictionary.items():
        new_key = key.replace('(', '').replace(')', '')
        renamed_dict[new_key] = value
    return renamed_dict


if __name__ == '__main__':

    # load the configuration file 
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # set the tracking uri 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # start mlflow experiment 
    with mlflow.start_run(run_name=params['name']):
        # load a pre-trained model 
        pre_trained_model = YOLO(params['model_type'])

        # train 
        model = pre_trained_model.train(
            data=data_yaml_path,
            imgsz=params['imgsz'],
            batch=params['batch'],
            epochs=params['epochs'],
            optimizer=params['optimizer'],
            lr0=params['lr0'],
            seed=params['seed'],
            pretrained=params['pretrained'],
            name=params['name']
        )

        # log params with mlflow
        mlflow.log_param('model_type', params['model_type'])
        mlflow.log_param('epochs',params['epochs'])
        mlflow.log_param('optimizer', params['optimizer'])
        mlflow.log_param('learning_rate', params['lr0'])

        # log metrics
        mlflow.log_metrics(rename_dict_keys(model.results_dict))
