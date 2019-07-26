import argparse
import glob
import os

import numpy as np
from scipy import ndimage
from PIL import Image
from tqdm import tqdm

def main(args):
    image_paths = glob.glob(os.path.join(args.annotation_train_dir, "*.png"))
    image_paths.extend(glob.glob(os.path.join(args.annotation_valid_dir, "*.png")))
    invalid_images = []
    for image_path in tqdm(image_paths):
        mask = Image.open(image_path).convert("L")
        
        # Convert mask to numpy array
        mask = np.array(mask)
        
        # Labels are decoded as different colors
        mask_labels = np.unique(mask)
        
        boxes = []
        
        for mask_label in mask_labels:
            # Ignore the background/unlabeled
            if mask_label == 255:
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
        if len(boxes) == 0:
            invalid_images.append(image_path)
    with open(args.output_file, "w") as f:
        f.write("\n".join(invalid_images))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test and generate invalid image list"
    )
    parser.add_argument("--annotation-train-dir",
                        default="../cocostuff/dataset/annotations/train2017",
                        help="Annotation directory")
    parser.add_argument("--annotation-valid-dir",
                        default="../cocostuff/dataset/annotations/val2017",
                        help="Annotation directory")
    parser.add_argument("--output-file",
                        default="invalid_images.txt",
                        help="Output file of invalid images")
    
    args = parser.parse_args()
    main(args)
    
    