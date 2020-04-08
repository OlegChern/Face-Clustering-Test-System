import datetime
import logging
import logging.config


class Logger:
    ConfigDir = "."
    Logger = None

    def __init__(self, log_config_dir=ConfigDir):
        current_date = datetime.datetime.now()
        current_date = current_date.strftime("%Y-%m-%d")

        dictLogConfig = {
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
                    "format": "%(asctime)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            }
        }

        logging.config.dictConfig(dictLogConfig)
        self.Logger = logging.getLogger("TestSystem")
        self.Logger.info("Logging Started")

    def info(self, message):
        self.Logger.info(message)
