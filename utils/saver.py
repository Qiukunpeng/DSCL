import os
import time
import datetime


class Saver(object):
    def __init__(self, args):
        """
            Initialize the parameters
        Args:
            args (argparse.Namespace): Arguments
        """
        self.args = args
        self.savers_folder_path = os.path.join(args.saver, args.dataset, args.model)


    def __call__(self):
        """
            Create saver path
            
        Returns:
            str: Path to saver folder
        """
        saver_folder_path = os.path.join(self.savers_folder_path, "experiment_" + str(self.args.experiment))

        if not os.path.exists(saver_folder_path):
            os.makedirs(saver_folder_path, exist_ok=True)
            
            return saver_folder_path
            
        # elif self.args.resume:
        #     return saver_folder_path
        
        else:
            folder = self._experiment_folder()
            serial_number = folder.split("_")[1]
            assert not os.path.exists(saver_folder_path), f"Now experiment_{int(serial_number) + 1} folder should be created!"
            
        
    def _experiment_folder(self):
        """
            Return experiment folder

        Returns:
            folders (str): Most recent created saver folder
        """
        # Get sub-folders in saver folder
        folders = os.listdir(self.savers_folder_path)

        folders = [f for f in folders if f.split("_")[0] == "experiment"]

        # Sort folders by experiment number
        folders = sorted(folders, key=lambda f: int(f.split("_")[1]))
        return folders[-1]

