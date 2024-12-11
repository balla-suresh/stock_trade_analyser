import os
import logging
from datetime import datetime
from pathlib import Path

class LoggerUtils:

    def __init__(self, log_file_suffix):
        ROOT_DIR = Path(__file__).parent.parent.parent.parent
        log_folder = f"{ROOT_DIR}/logs"

        self.log_file_name = datetime.now().strftime(
            f"{log_folder}/%Y-%m-%d-%H-%M-%S_{log_file_suffix}.log")

    def get_logger(self):
        os.makedirs(os.path.dirname(self.log_file_name), exist_ok=True)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d ] %(message)s',
            handlers=[logging.FileHandler(self.log_file_name)]
        )
        logging.getLogger(__name__).addHandler(logging.StreamHandler())
        return logger
    
