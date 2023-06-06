"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

#from bike_share.predict import make_prediction
from bike_share.pipeline import bike_share_pipe
from bike_share.processing.data_manager import get_year_and_month
from sklearn.metrics import mean_squared_error, r2_score


def test_make_prediction(sample_input_data):
    # Given
    # When
    print(sample_input_data[0])
    bike_share_pipe.fit(sample_input_data[0],sample_input_data[2]) 
    
    #result = make_prediction(input_data=sample_input_data[1])
    
    sample_input_data[1] = get_year_and_month(sample_input_data[1])
    
    y_pred = bike_share_pipe.predict(sample_input_data[1])
    r2_s = r2_score(sample_input_data[3])
    print("R2 score:", r2_s)
    print("Mean squared error:", mean_squared_error(sample_input_data[3], y_pred))

    # Then
    # r2_s = result.get("r2_score")
    #assert isinstance(predictions, np.ndarray)
    assert isinstance(r2_s, np.float64)
    # assert result.get("errors") is None
    assert r2_s > 0.8

