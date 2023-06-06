
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pdb

import numpy as np
from bike_share.config.core import config
from bike_share.processing.features import WeekdayImputer
from tests.conftest import sample_input_data

def test_Weekday_Is_Nan(sample_input_data):
    # Given
    assert np.isnan(sample_input_data.loc[5,'weekday'])
   

if __name__ == "__main__":
    X_Test = sample_input_data()
    test_Weekday_Is_Nan(X_Test)