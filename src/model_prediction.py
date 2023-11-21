import argparse
import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from config import Config
from data_monitoring import get_logger

PROJECT_PATH = Path(__file__).parent.resolve() / '../'
config = Config()
LOGGER = get_logger("Model_prediction")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a Pandas DataFrame.

    Parameters:
    - file_path (str): The relative path to the CSV file containing the data.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the loaded data.
    """

    LOGGER.info("[Load Data] Loading test data...")
    df = pd.read_csv(f"{PROJECT_PATH}/{file_path}")
    LOGGER.info("[Load Data] Test data successfully read.")

    return df


def load_model(model_path: str) -> TabularPredictor:
    """
    Load a pre-trained model from the specified path using the TabularPredictor.

    Parameters:
    - model_path (str): The relative path to the saved model.

    Returns:
    - TabularPredictor: A loaded TabularPredictor model.
    """

    LOGGER.info("[Load Data] Loading the model...")
    model = TabularPredictor.load(f"{PROJECT_PATH}/{model_path}")
    LOGGER.info("[Load Data] Model successfully got.")

    return model


def make_predictions(df: pd.DataFrame, model: TabularPredictor) -> dict:
    """
    Make predictions using a trained model on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which predictions will be made.
    - model: The trained model used for making predictions.

    Returns:
    - dict: A dictionary containing the predictions in the desired format.
    """

    LOGGER.info(f"[Make Predictions] Evaluating model over test: {model.evaluate(df, silent=True)}")
    LOGGER.info("[Make Predictions] Executing predictions...")

    result_df = df[['StartTime']].copy()
    result_df['Country_Super'] = model.predict(df)
    result_df['Country_ID'] = result_df['Country_Super'].map(config.data_processing["MAP_COUNTRY_ID"])

    # Crear un diccionario con el formato deseado a partir del DataFrame
    predictions = {"target": dict(zip(result_df.index + 1, result_df['Country_ID']))}

    return predictions


def save_predictions(predictions, predictions_file: str):
    """
    Save predictions to a JSON file.

    Parameters:
    - predictions: The predictions to be saved, typically in dictionary format.
    - predictions_file (str): The relative path to the file where predictions will be saved.
    """

    LOGGER.info(f"[Save Predictions] Writing the predictions in {PROJECT_PATH}/{predictions_file}")
    with open(f"{PROJECT_PATH}/{predictions_file}", 'w') as json_file:
        json.dump(predictions, json_file, indent=4)

    LOGGER.info("[Save Predictions] Predictions successfully saved.")


def parse_arguments():
    """
    Parse command-line arguments for the model prediction script.

    Returns:
    - argparse.Namespace: An object containing parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/test_data.csv',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='predictions/predictions.json',
        help='Path to save the predictions'
    )
    return parser.parse_args()


def main(input_file, model_file, output_file):
    LOGGER.info("Starting...")
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
