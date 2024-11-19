import pandas as pd

def dataframe_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed summary of a DataFrame's columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame for which to generate the summary.
        
    Returns:
        pd.DataFrame: A DataFrame containing information about each column, including:
            - Column: The name of the column
            - Data Type: The data type of the column
            - Unique Count: The number of unique values in the column
            - Unique Sample: A sample of unique values (up to 5)
            - Missing Values: The count of missing values in the column
            - Missing Percentage: The percentage of missing values in the column
    """
    report = pd.DataFrame(columns=['Column', 'Data Type', 'Unique Count', 'Unique Sample', 'Missing Values', 'Missing Percentage'])
    for column in df.columns:
        data_type = df[column].dtype
        unique_count = df[column].nunique()
        unique_sample = df[column].unique()[:5]
        missing_values = df[column].isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        report = pd.concat([report, pd.DataFrame({'Column': [column],
                                                      'Data Type': [data_type],
                                                      'Unique Count': [unique_count],
                                                      'Unique Sample': [unique_sample],
                                                      'Missing Values': [missing_values],
                                                      'Missing Percentage': [missing_percentage.round(4)]})],
                             ignore_index=True)
    return report