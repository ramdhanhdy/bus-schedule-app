# steps/data_ingestion_step.py

import os
from typing import List, Dict
import pandas as pd
from zenml import step
from src.data_ingestion import load_data

@step
def data_ingestion_step(file_info_list: List[Dict[str, str]]) -> pd.DataFrame:
    """
    ZenML step for data ingestion.
    
    Args:
        file_info_list (List[Dict[str, str]]): List of dictionaries containing file information.
            Each dictionary should have 'file_path' and 'file_type' keys, and optionally a 'correction_factor' key.

    Returns:
        pd.DataFrame: Combined DataFrame from all ingested files.
    """
    dfs = []
    for file_info in file_info_list:
        file_path = file_info['file_path']
        file_type = file_info['file_type']
        correction_factor = file_info.get('correction_factor')
        
        df = load_data(file_path, file_type=file_type, correction_factor=correction_factor)
        dfs.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

# Example usage (on pipeline)
# if __name__ == "__main__":
#     project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     file_info_list = [
#         {'file_path': os.path.join(project_dir, 'data', 'raw', 'location_data_Feb2024.csv'), 'file_type': 'csv'},
#         {'file_path': os.path.join(project_dir, 'data', 'raw', 'data_march_2024.csv'), 'file_type': 'csv'},
#         {'file_path': os.path.join(project_dir, 'data', 'raw', 'data_april.csv'), 'file_type': 'csv'},
#         {'file_path': os.path.join(project_dir, 'data', 'raw', 'location_data_May2024.csv'), 'file_type': 'csv'}
#     ]
    
#     result = data_ingestion_step(file_info_list)
#     print(result.head())
#     print(f"Total rows: {len(result)}")