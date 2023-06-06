import sys
sys.path.append('C:/Users/karna/Desktop/MLOps_M3_M4/project')

import typing as t
from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from bike_share import __version__ as _version
from bike_share.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

from bike_share.processing.features import OutlierHandler


##  Pre-Pipeline Preparation

# Extract year and month from the date column and create two another columns

def get_year_and_month(df):

    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()
    
    return df

numerical_features = []
categorical_features = []
target_col = [config.model_config.target]
unused_colms = config.model_config.unused_fields
outlier_remover = OutlierHandler(1.5)

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    for col in data_frame.columns:
        if col not in target_col + unused_colms:
            if data_frame[col].dtypes == 'float64':
                numerical_features.append(col)
            else:
                categorical_features.append(col)
    outlier_remover.fit(data_frame[numerical_features])
    outlier_remover.transform(data_frame[numerical_features])
    data_frame = get_year_and_month(data_frame)
    # drop unnecessary variables
    #data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
