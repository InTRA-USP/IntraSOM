
from logging.config import dictConfig
import matplotlib

dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
    "root": {
        "level": "NOTSET",
        "handlers": ["console"]
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "basic"
        }
    },
    "formatters": {
        "basic": {
            "format": '%(message)s'
        }
    }
})



from .intrasom import SOMFactory
