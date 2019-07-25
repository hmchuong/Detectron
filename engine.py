"""Training script for one epoch and Evaluation script
"""
import torch
import torch.nn as nn
import fastprogress
from fastprogress import master_bar, progress_bar

from utils import reduce_dict, log_loss, standardlize_masks
from coco_utils import get_coco_api_from_dataset
from cocostuff_eval import CocStuffEvaluator
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pycocotools.cocostuffeval import COCOStuffeval
from category import catdata

def train_one_epoch(model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    data_loader: torch.utils.data.DataLoader, 
                    master_progress_bar: master_bar,
                    device: str="cpu"):
    """Train model in one epoch
    
    Parameters
    ----------
    model: torch.nn.Module
        model to train
    optimizer: torch.optim.Optimizer
        optimize function
    data_loader: torch.utils.data.DataLoader
        dataset loader in batch
    device: str (default: "cpu")
        "cpu" or "cuda", device to train
    master_progress_bar: fastprogress.master_bar
        progress bar to update trainning information
    
    Returns
    -------
    float
        loss of current training epoch
    """
    
    # Switch model to training mode
    model.train()
    training_loss = 0  # Storing total loss
    
    # For each batch
    train_progress_bar = progress_bar(data_loader, parent=master_progress_bar)
    for batch, (images, targets) in enumerate(train_progress_bar):
        
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Back propagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Log loss
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        training_loss += losses_reduced.item()

        mean_loss = training_loss / (batch + 1)
        other_loss = log_loss(loss_dict_reduced)
        log = "Mean: %.2f %s" % (mean_loss, other_loss)
        master_progress_bar.child.comment = log

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Return training loss
    return training_loss / len(data_loader)


def evaluate(model: torch.nn.Module, 
             data_loader: torch.utils.data.DataLoader, 
             master_progress_bar: fastprogress.master_bar,
             device: str="cpu"):

    model.eval()
    coco_gt = get_coco_api_from_dataset(data_loader.dataset)
    coco_pred = COCO()
    
    ann_id = 0
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    
    with torch.no_grad():  # Turn off gradient
        
        # For each batch
        for batch, (images, targets) in enumerate(progress_bar(data_loader, parent=master_progress_bar)):
            # Move images and targets to device
            new_images = list(image.to(device) for image in images)
            
            # Predict outputs
            outputs = model(new_images)
            
            # Move to CPU for evaluation
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

            for img, target, output in zip(images, targets, outputs):
                image_id = target["image_id"].item()
                img_dict = {}
                img_dict["id"] = image_id
                img_dict['height'] = img.shape[-2]
                img_dict['width'] = img.shape[-1]
                dataset['images'].append(img_dict)
                bboxes = target["boxes"]
                bboxes[:, 2:] -= bboxes[:, :2]
                areas = (bboxes[:, 3] * bboxes[:, 2]).tolist()
                bboxes = bboxes.tolist()
                labels = target['labels'].tolist()
                iscrowd = [0] * len(labels)
                masks = target['masks']
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
                masks = standardlize_masks(masks)
                
                num_obj = len(bboxes)
                for i in range(num_obj):
                    label = labels[i]
                    if labels[i] > 182 or labels[i] < 92:
                        label = 183
                    ann = {}
                    ann['image_id'] = image_id
                    ann['bbox'] = bboxes[i]
                    ann['category_id'] = label
                    ann['area'] = areas[i]
                    ann['iscrowd'] = iscrowd[i]
                    ann['id'] = ann_id
                    
                    mask = masks[i]
                    mask[mask == 0] = 183
                    ann['segmentation'] = coco_mask.encode(mask)
                    
                    dataset['annotations'].append(ann)
                    
                    ann_id += 1
        dataset['categories'] = catdata
        coco_pred.dataset = dataset
        coco_pred.createIndex()
    import time
    before = time.clock()
    coco_eval = COCOStuffeval(coco_gt, coco_pred, stuffStartId=92, stuffEndId=182, addOther=True)
    coco_eval.evaluate()
    coco_eval.summarize()
    after = time.clock()
    print('Evaluation took %.2fs!' % (after - before))
    
    return coco_eval
    
            
        
