import logging


class Logger(logging.Logger):

    def __init__(self, level: int = logging.INFO):
        super().__init__(name="gdp_duacs_jpw", level=level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)


LOGGER = Logger()
