import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class MyDataset(data.Dataset):
    def __init__(self, args, root, logger, split="train", transform=True):
        """
            Initialize the parameters
        Args:
            root (str): The root of the dataset
            logger (logging.Logger): Logger
            split (str): The split of the dataset
            transform (torchvision.transforms): The transform of the dataset
            is_cms (bool optional): Whether to calculate the mean and std of the dataset
        """
        self.args = args
        self.root = root
        self.logger = logger
        self.split = split
        self.interpolation = InterpolationMode(self.args.interpolation)
        self.transform = transform

        with open(os.path.join(self.root, self.split, self.split + ".txt"), "r") as f:  # Open the txt file
            self.image_info = f.readlines()  # Read all the lines in the txt file
        self.data = [line.strip().split(" ") for line in self.image_info]  # Get the image path and label

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Set the mean and std from ImageNet

        self.logger.info(f"mean: {self.mean}, std: {self.std}")

    def __getitem__(self, index):
        """
            Get the image and label of the dataset
        Args:
            index (int): The index of the dataset

        Returns:
            image (torch.Tensor): The image of the dataset
            label (torch.Tensor): The label of the dataset
        """
        image_path, label = self.data[index]  # Get the image path and label
        image = Image.open(image_path).convert("RGB")  # Open the image

        if self.transform is not None:  # If the transform is not None
            image = self.transform_image(image)  # Transform the image

        label = int(label)  # Convert the label to int

        return image, label  # Return the image and label

    def __len__(self):
        """
            Get the length of the dataset

        Returns:
            length (int): The length of the dataset
        """
        return len(self.data)  # Return the length of the dataset

    def transform_image(self, image):
        """
            Transform the image
        Args:
            image (PIL.Image): The image of the dataset

        Returns:
            image (torch.Tensor): The transformed image
        """
        if self.split == "train":  # If the split is "train"
            transform_methods = [
                transforms.RandomResizedCrop(size=self.args.train_size, scale=(0.8, 1.0), 
                                             ratio=(4. / 4., 4. / 4.), interpolation=self.interpolation),  # Random crop the image to 224 * 224 with a random scale and ratio
                transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally, the probability is 0.5
                transforms.ToTensor(),  # Transform the image to tensor 
                transforms.Normalize(mean=self.mean, std=self.std),  # Normalize the image
            ]
            composed_transforms = transforms.Compose(transform_methods)
        else:  # If the split is not "train"
            transform_methods = [
                transforms.Resize(size=(self.args.val_size, self.args.val_size), interpolation=self.interpolation),  # Resize the image to 224 * 224
                transforms.ToTensor(),  # Transform the image to tensor
                transforms.Normalize(mean=self.mean, std=self.std)  # Normalize the image
            ]
            composed_transforms = transforms.Compose(transform_methods)
        return composed_transforms(image)  # Return the transformed image