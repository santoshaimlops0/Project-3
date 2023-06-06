import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from bike_share.config.core import config
from bike_share.pipeline import bike_share_pipe
from bike_share.processing.data_manager import load_dataset, save_pipeline
# from bike_share.processing.data_manager import get_year_and_month
# from sklearn.metrics import mean_squared_error, r2_score



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

    # Pipeline fitting
    bike_share_pipe.fit(X_train,y_train)  #
    # X_test = get_year_and_month(X_test)
    
    # y_pred = bike_share_pipe.predict(X_test)
    # print("R2 score:", r2_score(y_test, y_pred))
    # print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist= bike_share_pipe)
    
    # printing the score
    
if __name__ == "__main__":
    run_training()