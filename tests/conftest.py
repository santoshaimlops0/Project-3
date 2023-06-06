import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from sklearn.model_selection import train_test_split

from bike_share.config.core import config
from bike_share.processing.data_manager import _load_raw_dataset, load_dataset

# from bike_share.predict import split_data


@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_config.training_data_file)
    #data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data,  # predictors
    #     data[config.model_config.target],
    #     test_size=config.model_config.test_size,
    #     # we are setting the random seed here
    #     # for reproducibility
    #     random_state=config.model_config.random_state,
    # )
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    
    # X_test, y_test = split_data(data)
    
    #return [X_train, X_test, y_train, y_test]
    return X_test