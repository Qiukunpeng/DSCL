import os
import logging


class Logger(object):
    def __init__(self, args, saver_folder_path):
        """
            Initialize the parameters
        Args:
            args (argparse.Namespace): Arguments
            saver_folder_path (str): Path to saver folder
        """
        self.args = args
        self.saver_folder_path = saver_folder_path

        self.log_folder_path = os.path.join(self.saver_folder_path, self.args.log)
        if not os.path.exists(self.log_folder_path):
            os.makedirs(self.log_folder_path, exist_ok=True)

    def __call__(self, verbosity=1, name=None):
        """
            Return logger
        Args:
            verbosity (int): Verbosity level
            name (str): Name of logger

        Returns:
            logger (logging.Logger): Logger
        """
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")

        logger = logging.getLogger(name)
        # logger.setLevel(level=level_dict[verbosity] if self.args.rank in [-1, 0] else level_dict[(verbosity + 1)])
        logger.setLevel(level=level_dict[verbosity])

        log_file_path = os.path.join(self.log_folder_path, "logs.txt")
        # If we want to overwrite the past logs, we have to set the mode to "w"
        fh = logging.FileHandler(log_file_path)
        # We can set the level of each Handler to filter out low-level logs
        # logger.setLevel(level=level_dict[verbosity] if self.args.rank in [-1, 0] else level_dict[(verbosity + 1)])
        logger.setLevel(level=level_dict[verbosity])
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        # logger.setLevel(level=level_dict[verbosity] if self.args.rank in [-1, 0] else level_dict[(verbosity + 1)])
        logger.setLevel(level=level_dict[verbosity])
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger
