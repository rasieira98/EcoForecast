from pathlib import Path
from typing import Any

import yaml

RESOURCES_PATH = Path(__file__).parent.resolve() / '../resources'
CONFIG_FILE_PATH = f"{RESOURCES_PATH}/config.yaml"


class Config:
    """
    Configuration class for managing data processing and training parameters.

    Attributes:
    - data_processing (dict[str, Any]): Dictionary containing data processing parameters.
    - training (dict[str, Any]): Dictionary containing training parameters.
    """

    data_processing: dict[str, Any]
    training: dict[str, Any]

    def __init__(self):
        with open(CONFIG_FILE_PATH, 'r') as file:
            data = yaml.safe_load(file)

        self.data_processing = data["data_processing"]
        self.training = data["training"]
