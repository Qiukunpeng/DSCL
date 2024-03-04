import os


class Convert2Txt(object):
    def __init__(self, root="./", split="train"):
        """
            Initialize the parameters

        Args:
            root (str): The root directory of the dataset
            split (str): The split of the dataset
        """
        self.root = root
        self.split = split

    def txt(self):
        """
            Convert the dataset to txt format

        Returns:
            None
        """
        root_dir = os.path.join(self.root, self.split)  # Get the root directory
        class_names = sorted(os.listdir(root_dir))  # Get all the class names
        with open(os.path.join(self.root, self.split, self.split + ".txt"), "w") as f:  # Open the txt file
            for index, label in enumerate(class_names):
                for image_name in sorted(os.listdir(os.path.join(root_dir, label))):
                    path_convert = os.path.join(self.root, self.split, label, image_name)
                    f.write(path_convert + " " + str(index) + "\n")  # Write the path and label to the txt file


if __name__ == "__main__":
    # ---------------------------Set variables-------------------------- #
    root = "/opt/data/private/QiuKunpeng/Classification/DSCL/data/Fold-0"
    # --------------------Test the Convert2Txt class------------------- #
    for split in ["train", "val"]:  # For train and val
        Convert2Txt(root, split).txt()  # Convert the dataset to txt file
