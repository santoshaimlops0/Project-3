import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from bike_share import __version__ as _version
from bike_share.config.core import config
from bike_share.pipeline import bike_share_pipe
from bike_share.processing.data_manager import load_pipeline
from bike_share.processing.data_manager import get_year_and_month
from bike_share.processing.data_manager import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bike_share_pipe= load_pipeline(file_name=pipeline_file_name)

y_test1 = pd.DataFrame
def make_prediction(*,input_data) -> dict:
    """Make a prediction using a saved model """
    data_frame=pd.DataFrame(input_data[0])
    data = get_year_and_month(data_frame)
    #data=data.reindex(columns=['dteday','season','hr','holiday', 'weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered','yr','mnth'])
    
    y_pred = bike_share_pipe.predict(data)
    r2_s = r2_score(input_data[1], y_pred)
    print("Mean squared error:", mean_squared_error(input_data[1], y_pred))
    results = {"r2_score": r2_s, "version": _version, "error": None }
    print(results)

    return results

def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    
    return X_test, y_test
if __name__ == "__main__":
    data = load_dataset(file_name=config.app_config.training_data_file)
    X_test, y_test = split_data(data)
    #y_test1 = y_test
    make_prediction(input_data=[X_test,y_test])