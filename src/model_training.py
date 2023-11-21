import argparse
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from config import Config
from data_monitoring import get_logger

PROJECT_PATH = Path(__file__).parent.resolve() / '../'
MODEL_PATH = f"{PROJECT_PATH}/models"

config = Config()
LOGGER = get_logger("Model_training")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a Pandas DataFrame.

    Parameters:
    - file_path (str): The relative path to the CSV file containing the data.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the loaded data.
    """
    LOGGER.info("[Load Data] Loading training data...")
    df = pd.read_csv(f"{PROJECT_PATH}/{file_path}")
    LOGGER.info("[Load Data] Training data successfully read.")

    return df


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the input DataFrame into training and testing sets, save the sets as CSV files,
    and return the training set.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be split.

    Returns:
    - pd.DataFrame: The training set as a Pandas DataFrame.
    """

    LOGGER.info(f"[Split Data] Spliting 80% train and 20% test...")
    df_train, df_test = train_test_split(df,
                                         test_size=config.training["test_size"],
                                         random_state=config.training["random_state"])
    LOGGER.info(f"[Split Data] Size train = {df_train.shape[0]}; Size test = {df_test.shape[0]}")

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    LOGGER.info(f"[Save data] Saving train data in {PROJECT_PATH}/data/train_data.csv")
    df_train.to_csv(f"{PROJECT_PATH}/data/train_data.csv", index=False)
    LOGGER.info(f"[Save data] Saving train data in {PROJECT_PATH}/data/test_data.csv")
    df_test.to_csv(f"{PROJECT_PATH}/data/test_data.csv", index=False)

    return df_train


def train_and_save_model(df_train: pd.DataFrame, model_path: str):
    """
    Train a model using the TabularPredictor from the `autogluon.tabular` module,
    and save the trained model to the specified path.

    Parameters:
    - df_train (pd.DataFrame): The training set as a Pandas DataFrame.
    - model_path (str): The relative path where the trained model should be saved.
    """

    LOGGER.info(f"[Train Model] Starting the training. After the training the model is automatically saved.")
    (
        TabularPredictor(label='target', eval_metric=config.training["eval_metric"],
                         path=f"{PROJECT_PATH}/{model_path}")
        .fit(df_train, holdout_frac=config.training["holdout_frac"], presets=config.training["presets"])
    )


def parse_arguments():
    """
    Parse command-line arguments for the model training script.

    Returns:
    - argparse.Namespace: An object containing parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/processed_data.csv',
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='models/model.pkl',
        help='Path to save the trained model'
    )
    return parser.parse_args()


def main(input_file, model_file):
    LOGGER.info("Starting...")
    df = load_data(input_file)
    df_train = split_data(df)
    train_and_save_model(df_train, model_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)
