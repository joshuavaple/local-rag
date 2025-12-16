import logging
from rich.logging import RichHandler


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # default Python logger
        # handler = logging.StreamHandler()
        # fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        # handler.setFormatter(fmt)
        # logger.addHandler(handler)

        # rich logger for colors
        handler = RichHandler(
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
            log_time_format="%Y-%m-%d %H:%M:%S"  # <-- ISO-style timestamp
        )
        fmt = "%(message)s" 
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
