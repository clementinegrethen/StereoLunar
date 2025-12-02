from collections import defaultdict
import logging


def logging_level_from_str(logging_level_str: str) -> int:
    """
    Args:
        logging_level_str (str): DEBUG|INFO|WARNING|ERROR
    """
    mappings = defaultdict(lambda: logging.DEBUG)
    mappings.update(
        {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
    )
    return mappings[logging_level_str]


def get_logger(name: str, logging_level: str = "DEBUG") -> logging.Logger:
    """
    Args:
        name (str): logger name, you may use `__name__` or `__file__` to name your logger
        logging_level (str): DEBUG|INFO|WARNING|ERROR logging level for this logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging_level_from_str(logging_level))
    try:
        import coloredlogs

        coloredlogs.install(level=logging_level, logger=logger)
    except ImportError:
        logger.info(
            "coloredlogs not installed, logs will not be colored, (you can install using 'pip install coloredlogs')"
        )
    return logger


def test_get_logger():
    logger = get_logger(__file__, logging_level="DEBUG")
    logger.debug("debug msg")
    logger.info("info msg")
    logger.warning("warning msg")
    logger.error("error msg")
