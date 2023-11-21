import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from config import Config
from data_monitoring import get_logger

PROJECT_PATH = Path(__file__).parent.resolve() / '../'

LOGGER = get_logger("data_ingestion")
config = Config()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a Pandas DataFrame.

    Parameters:
    - file_path (str): The relative path to the CSV file containing the data.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the loaded data.
    """

    LOGGER.info(f"[Load data] Reading file {file_path}...")
    df = pd.read_csv(f"{PROJECT_PATH}/{file_path}")
    LOGGER.info(f"[Load data] File read successfully")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the input DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.

    1. Convert the 'StartTime' column to datetime format.
    2. Sort the DataFrame based on the 'StartTime' column.
    3. Filter out rows where 'AreaID' is null, logging the number of records lost.
    4. Filter columns based on the configuration's "columns_clean" parameter.
    5. Remove outliers using the 'remove_outliers' function.
    """

    LOGGER.info(f"[Clean data] Starting the process...")
    LOGGER.info(f"[Clean data] Nº records before the cleaning: {df.shape[0]}")

    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M+00:00Z')
    df_clean = df.sort_values('StartTime').reset_index(drop=True)

    LOGGER.info(f"[Clean data] Filtering AreaID != null")
    df_clean = df_clean[df_clean['AreaID'].notna()].reset_index(drop=True)
    LOGGER.info(f"[Clean data] After filtering AreaID != null we lose {df.shape[0] - df_clean.shape[0]} records")

    LOGGER.info(f"[Clean data] Filtering by columns which are renewable")
    df_clean = df_clean.filter(items=config.data_processing["columns_clean"])

    LOGGER.info(f"[Clean data] Removing outliers...")
    df_clean = df_clean.groupby('Country', group_keys=True).apply(remove_outliers).reset_index(drop=True)

    LOGGER.info(f"[Clean data] Nº records after the cleaning: {df_clean.shape[0]} "
                f"(Total lost = {df.shape[0] - df_clean.shape[0]})")
    LOGGER.info(f"[Clean data] Process completed.")

    return df_clean


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing on the input DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be preprocessed.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.

    1. Interpolate missing values by interval for each country.
    2. Resample the DataFrame to hourly frequency.
    3. Create columns 'green_energy' and 'surplus' based on specified calculations.
    4. Create a target table with columns 'StartTime' and 'target'.
    5. Pivot the final table based on specified columns.
    6. Merge the pivot table with the target table.

    """
    LOGGER.info(f"[Preprocess data] Starting the process...")
    LOGGER.info(f"[Preprocess data] Nº records before processing: {df.shape[0]}")

    LOGGER.info(f"[Clean data] Treating nan values by interpolation by interval")
    date_min = df["StartTime"].min()
    date_max = df["StartTime"].max()
    df_processed = df.groupby('Country').apply(
        lambda x: interpolate_interval(x, date_min, date_max)).reset_index(drop=True)

    LOGGER.info(f"[Preprocess data] Executing resampling by hour")
    df_processed = df_processed.groupby('Country').apply(resample_h)
    country_series = df_processed.index.get_level_values('Country')
    df_processed.reset_index(drop=True, inplace=True)
    df_processed['Country'] = country_series
    LOGGER.info(f"[Preprocess data] Nº records after resampling: {df_processed.shape[0]}")

    LOGGER.info(f"[Preprocess data] Creating the columns green_energy and surplus...")
    df_processed['green_energy'] = df_processed.filter(regex='^B').sum(axis=1, min_count=1)
    df_processed['surplus'] = df_processed['green_energy'] - df_processed['Load']

    LOGGER.info(f"[Preprocess data] Creating the target table (StartTime, target)")
    target_df = calc_target_df(df_processed)

    LOGGER.info(f"[Preprocess data] Pivoting the final table...")
    df_pivot = pd.pivot_table(df_processed, index=["StartTime"], columns=["Country"],
                              values=config.data_processing["columns_pivot"], dropna=False)
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_processed = df_pivot.reset_index()

    LOGGER.info(f"[Preprocess data] Merging pivot table with target table")
    df_final = (
        pd.merge(df_processed, target_df, on='StartTime')
        .sort_values(['StartTime'])
        .reset_index(drop=True)
    )
    num_row_no_na = df_final.dropna(subset=[col for col in df_final.columns if col.startswith("surplus")])
    LOGGER.info(f"[Preprocess data] Values with all rows without a null value: {num_row_no_na.shape[0]}")
    LOGGER.info(f"[Preprocess data] Nº records after the processing: {df_final.shape[0]} "
                f"(Total lost = {df.shape[0] - df_final.shape[0]})")
    LOGGER.info(f"[Preprocess data] Value counts target: {df_final['target'].value_counts()}")
    LOGGER.info(f"[Preprocess data] Process completed.")

    return df_final


def calc_target_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the target DataFrame based on the input DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which the target will be calculated.

    Returns:
    - pd.DataFrame: The target DataFrame containing 'StartTime' and 'target' columns.
    """
    df_target = df.dropna(subset=["surplus"])
    idx_max_surplus = df_target.groupby('StartTime')['surplus'].idxmax()
    df_target = df_target.loc[idx_max_surplus][['StartTime', 'Country']].reset_index(drop=True)
    df_target['StartTime'] = df_target['StartTime'] - pd.Timedelta(hours=1)  # Target next time
    df_target.columns = ['StartTime', 'target']

    return df_target


def interpolate_interval(df: pd.DataFrame, date_min, date_max) -> pd.DataFrame:
    """
    Interpolate missing values in the input DataFrame for specified columns and time interval.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be interpolated.
    - date_min: The minimum date for the interpolation interval.
    - date_max: The maximum date for the interpolation interval.

    Returns:
    - pd.DataFrame: The DataFrame with interpolated values.
    """

    columns = [col for col in df.columns if col.startswith('Load') or col.startswith('B')]
    country = df["Country"].to_list()[0]

    # Create a placeholder dataframe with all the timestamps
    date_range = pd.date_range(start=date_min, end=date_max,
                               freq=config.data_processing["country_frequency"][country])
    df_full_range = pd.DataFrame({'StartTime': date_range})

    df = pd.merge(df_full_range, df, on='StartTime', how='left')
    df["Country"] = df["Country"].fillna(country)
    df["Date"] = df['StartTime'].dt.date
    df["Hour"] = df['StartTime'].dt.hour

    for column in columns:
        df[column] = df.groupby(["Date", "Hour"])[column].transform(
            lambda x: x.interpolate(method=config.data_processing["interpolate_method"],
                                    limit_direction=config.data_processing["interpolate_limit_direction"]))

    df.drop(columns=["Date", "Hour"], axis=1, inplace=True)

    LOGGER.info(f"Interpolate interval finished for {country}")
    return df


def resample_h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the input DataFrame to hourly frequency by summing numeric values.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be resampled.

    Returns:
    - pd.DataFrame: The resampled DataFrame with hourly frequency.
    """
    df.set_index('StartTime', inplace=True)
    df = df.resample('H').sum(numeric_only=True, min_count=1).reset_index()

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the input DataFrame based on Z-scores.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which outliers will be removed.

    Returns:
    - pd.DataFrame: The DataFrame with outliers set to null.
    """

    columns = [col for col in df.columns if col.startswith('Load') or col.startswith('B')]
    df.sort_values('StartTime', inplace=True)
    # Calculate Z-scores for each column
    z_scores = np.abs(zscore(df[columns]))

    # Define a threshold for Z-score (e.g., 3 standard deviations)
    threshold = config.data_processing["threshold_zscore"]

    # Identify outliers and set them to null
    outliers_mask = z_scores > threshold
    df[columns] = np.where(outliers_mask, np.nan, df[columns])
    outliers_count = np.sum(outliers_mask, axis=0)
    outliers_count_col = ','.join([f"{col}={outliers_count[col]}" for col in columns if outliers_count[col] > 0])

    if outliers_count_col:
        LOGGER.info(f"[Clean data] Outliers founds for {df['Country'].values[0]} = ({outliers_count_col})")
    else:
        LOGGER.info(f"[Clean data] No outliers found for {df['Country'].values[0]}")

    return df


def save_data(df, output_file):
    """
    Save the processed DataFrame to a CSV file.

    Parameters:
    - df: The DataFrame to be saved.
    - output_file (str): The relative path to the output CSV file.
    """

    LOGGER.info(f"[Save data] Saving the processing result in {PROJECT_PATH}/{output_file}")
    df.to_csv(f"{PROJECT_PATH}/{output_file}", index=False)
    LOGGER.info(f"[Save data] Process completed.")


def parse_arguments():
    """
    Parse command-line arguments for the data processing script.

    Returns:
    - argparse.Namespace: An object containing parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/processed_data.csv',
        help='Path to save the processed data'
    )
    return parser.parse_args()


def main(input_file, output_file):
    LOGGER.info("Starting...")
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)
