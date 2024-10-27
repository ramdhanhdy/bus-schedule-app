# src/data_ingestion.py

import pandas as pd
from abc import ABC, abstractmethod

class DataIngestionStrategy(ABC):
    @abstractmethod
    def ingest_data(self, file_path, **kwargs):
        pass

class CSVIngestionStrategy(DataIngestionStrategy):
    def ingest_data(self, file_path, **kwargs):
        return pd.read_csv(file_path)

class ExcelIngestionStrategy(DataIngestionStrategy):
    def ingest_data(self, file_path, **kwargs):
        df = pd.read_excel(file_path)
        correction_factor = kwargs.get('correction_factor')
        if correction_factor:
            df['latitude'] = df['latitude'] / correction_factor
            df['longitude'] = df['longitude'] / correction_factor
        return df

class DataIngestionFactory:
    @staticmethod
    def get_strategy(file_type):
        if file_type == 'csv':
            return CSVIngestionStrategy()
        elif file_type == 'excel':
            return ExcelIngestionStrategy()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

def load_data(file_path, file_type='csv', **kwargs):
    strategy = DataIngestionFactory.get_strategy(file_type)
    return strategy.ingest_data(file_path, **kwargs)