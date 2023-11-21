import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with the specified name and logging level.

    Parameters:
    - name (str): The name of the logger.
    - level (int): The logging level (default: logging.INFO).

    Returns:
    - logging.Logger: A configured logger object.
    """

    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level)

    return logger
