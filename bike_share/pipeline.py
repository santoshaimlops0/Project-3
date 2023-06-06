import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bike_share.config.core import config
from bike_share.processing.features import WeekdayImputer
from bike_share.processing.features import Mapper
from bike_share.processing.features import WeekdayOneHotEncoder
from bike_share.processing.features import ColumnDropperTransformer
from bike_share.processing.features import WeathersitImputer

bike_share_pipe=Pipeline([
    
    ("weekday_imputation", WeekdayImputer(variables=config.model_config.weekday_var)
    ),
    ("weathersit_imputation", WeathersitImputer(variables=config.model_config.weather_var)
    ),
    ##==========Mapper======##
    ("map_mnth",Mapper(config.model_config.mnth_var, config.model_config.mnth_mapping)
    ),
    ("map_season",Mapper(config.model_config.season_var, config.model_config.season_mapping )
    ),
    ("map_weather",Mapper(config.model_config.weather_var, config.model_config.weather_mapping)
    ),
    ("map_holiday",Mapper(config.model_config.holiday_var, config.model_config.holiday_mapping)
    ),
    ("map_workingday",Mapper(config.model_config.workingday_var, config.model_config.workingday_mapping)
    ),
    ("map_hour",Mapper(config.model_config.hour_var, config.model_config.hour_mapping)
    ),
    ("weekday_encoder", WeekdayOneHotEncoder(config.model_config.weekday_var)
    ),
    ("column_dropper", ColumnDropperTransformer(config.model_config.unused_fields)
    ),
    # scale
    ("scaler", StandardScaler()),
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
          
  ])