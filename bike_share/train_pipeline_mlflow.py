import sys
import warnings

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

from bike_share.config.core import config
from bike_share.pipeline import bike_share_pipe
from bike_share.processing.data_manager import load_dataset, save_pipeline
from bike_share.processing.data_manager import get_year_and_month
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from urllib.parse import urlparse

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    
    with mlflow.start_run():
        # Pipeline fitting
        print("111111")
        bike_share_pipe.fit(X_train,y_train)  #
        X_test = get_year_and_month(X_test)
    
        y_pred = bike_share_pipe.predict(X_test)
        print("22222")

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
                
        predictions = bike_share_pipe.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                bike_share_pipe, "model", registered_model_name="RandomForestRegressor", signature=signature
            )
        else:
            mlflow.sklearn.log_model(bike_share_pipe, "model", signature=signature)
    # print("R2 score:", r2_score(y_test, y_pred))
    # print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist= bike_share_pipe)
    
    # printing the score
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    run_training()