"""Define dataset architecture
"""
import os
import glob
import random

import numpy as np
from scipy import ndimage
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class COCOStuffDataset(object):
    """Define COCO-Stuff dataset schema
    
    Parameters
    ----------
    image_dir: str
        path to image directory (*.jpg)
    annotation_dir: str
        path to annotation directory (*.png)
    label_indices: list of int
        list of label indices
    background_index: int
        index of unlabeled/ background
    """
    def __init__(self, image_dir: str, annotation_dir: str, label_indices: list=list(range(182)), background_index: int=255):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) # Get all .jpg files
        self.annotation_dir = annotation_dir
        self.label_indices = label_indices
        self.background_index = background_index
        
    def __len__(self):
        """Length of the dataset
        """
        return len(self.image_paths) - len(self.invalid_images)
    
    def __getitem__(self, index):
        """Get item at index
        
        Parameters
        ----------
        index: int
            index of data
        """
        # Load image and mask
        image_path = self.image_paths[index]
        mask_path = os.path.join(self.annotation_dir, image_path.split('/')[-1].replace(".jpg", ".png"))
        img = Image.open(image_path).convert("RGB")
        img = F.to_tensor(img)
        mask = Image.open(mask_path).convert("L")
        
        # Convert mask to numpy array
        mask = np.array(mask)
        
        # Labels are decoded as different colors
        mask_labels = np.unique(mask)
        
        boxes = []
        masks = []
        labels = []
        
        for mask_label in mask_labels:
            # Ignore the background/unlabeled
            if mask_label == self.background_index:
                continue
            
            # Extract the mask of the current label
            independent_mask = mask == mask_label
            
            # Extract instance in the current mask
            blobs, no_blobs = ndimage.label(independent_mask)
            
            # For each instance
            for i in range(1, no_blobs + 1):
                # Get bounding box
                pos = np.where(blobs == i)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if (xmax - xmin) * (ymax - ymin) == 0:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                
                # Get instance mask
                instance_mask = (blobs == i)
                masks.append(instance_mask)
                
                # Add label
                if mask_label not in self.label_indices: print(mask_label)
                labels.append(mask_label + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.as_tensor([], dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img, target
        
            
            
            
