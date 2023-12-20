# import os
import logging
import logging.config
# import yaml

from pydantic import BaseModel


def setup_logger(logger_name: str, logging_level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(process)d | %(name)s | %(funcName)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S")

    logger_obj = logging.getLogger(name=logger_name)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(process)d | %(name)s | %(funcName)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    # ch.setLevel(log_level)

    logger_obj.addHandler(ch)
    logger_obj.setLevel(logging_level)
    logger_obj.propagate = False
    # logger_obj.addFilter()

    # uvicorn_logger = logging.getLogger("uvicorn")
    # uvicorn_logger.propagate = False

    return logger_obj


class LogConfig(BaseModel):
    """
    Logging configuration to be set for the server
    """

    LOGGER_NAME: str = "mycoolapp"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    # Logging config
    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }


# def read_logging_config(default_path="logging.yml", env_key="LOG_CFG"):
#     path = default_path
#     value = os.getenv(env_key, None)
#
#     if value:
#         path = value
#
#     if os.path.exists(path):
#         with open(path, "rt") as f:
#             logging_config = yaml.safe_load(f.read())
#         return logging_config
#     else:
#         return None
#
#
# def setup_logging(logging_config, default_level=logging.INFO):
#     if logging_config:
#         logging.config.dictConfig(logging_config)
#     else:
#         logging.basicConfig(level=default_level)
#
#
# class NoHealthCheckFilter(logging.Filter):
#     def __init__(self, param=None):
#         self.param = param
#
#     def filter(self, record):
#         return record.getMessage().find('/healthcheck') == -1
