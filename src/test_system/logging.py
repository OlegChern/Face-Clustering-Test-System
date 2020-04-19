import sys
import datetime
import logging
import logging.config


def get_file_logger(log_config_dir="./logs"):
    current_date = datetime.datetime.now()
    current_date = current_date.strftime("%Y-%m-%d")

    dict_log_config = {
        "version": 1,
        "handlers": {
            "fileHandler": {
                "class": "logging.FileHandler",
                "formatter": "myFormatter",
                "filename": f"{log_config_dir}/{current_date}.log"
            }
        },
        "loggers": {
            "TestSystem": {
                "handlers": ["fileHandler"],
                "level": "INFO",
            }
        },
        "formatters": {
            "myFormatter": {
                "format": "%(asctime)s\n%(message)s\n",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        }
    }

    logging.config.dictConfig(dict_log_config)

    return logging.getLogger("TestSystem")


def get_default_logger(name="Default"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s\n%(message)s\n')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
