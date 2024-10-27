# src/data_processing.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from src.data_ingestion import load_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessingStrategy(ABC):
    @abstractmethod
    def process(self, df):
        pass

# Data Preparation Strategies
class DropColumnsStrategy(DataProcessingStrategy):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def process(self, df):
        return df.drop(self.columns_to_drop, axis=1)

class ConvertToNumericStrategy(DataProcessingStrategy):
    def __init__(self, columns_to_convert):
        self.columns_to_convert = columns_to_convert

    def process(self, df):
        def convert_to_numeric(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return x
            elif isinstance(x, str):
                try:
                    # Remove all commas and then convert to float
                    return float(x.replace(',', ''))
                except ValueError:
                    logger.warning(f"Could not convert value '{x}' to numeric")
                    return np.nan
            else:
                logger.warning(f"Unexpected type {type(x)} for value {x}")
                return np.nan

        for column in self.columns_to_convert:
            df[column] = df[column].apply(convert_to_numeric)
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column {column} has {nan_count} NaN values after conversion")
        return df

class CorrectCoordinatesStrategy(DataProcessingStrategy):
    def __init__(self, correction_factor):
        self.correction_factor = correction_factor

    def process(self, df):
        if self.correction_factor:
            df['latitude'] = df['latitude'] / self.correction_factor
            df['longitude'] = df['longitude'] / self.correction_factor
        return df

class ConvertTimeStrategy(DataProcessingStrategy):
    def process(self, df):
        def parse_datetime(x):
            try:
                return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    return pd.to_datetime(x, format='%m/%d/%Y %H:%M', dayfirst=False)
                except ValueError:
                    logger.warning(f"Could not parse datetime: {x}")
                    return pd.NaT

        df['time'] = df['time'].apply(parse_datetime)
        return df.sort_values(by=['route_id', 'time'])

class ExtractTimeFeatureStrategy(DataProcessingStrategy):
    def process(self, df):
        df['hour'] = df['time'].dt.hour
        df['weekday'] = df['time'].dt.weekday
        df['day_name'] = df['time'].dt.day_name()
        df['month'] = df['time'].dt.month
        df['month_name'] = df['month'].apply(lambda x: pd.to_datetime(f"2024-{x}-01").strftime('%B'))
        return df

# Data Processing Strategies
class RemoveWeekendData(DataProcessingStrategy):
    def process(self, df):
        return df[df['weekday'].isin([0, 1, 2, 3, 4])]

class DataProcessor:
    def __init__(self, strategies):
        self.strategies = strategies

    def process(self, df):
        for strategy in self.strategies:
            df = strategy.process(df)
            # Log the number of NaN values in each column after each strategy
            for column in df.columns:
                nan_count = df[column].isna().sum()
                if nan_count > 0:
                    logger.info(f"After {strategy.__class__.__name__}, column {column} has {nan_count} NaN values")
        return df