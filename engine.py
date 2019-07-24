"""Training script for one epoch and Evaluation script
"""
import torch
import torch.nn as nn
import fastprogress
from fastprogress import master_bar, progress_bar

from utils import reduce_dict, log_loss
from coco_utils import get_coco_api_from_dataset
from cocostuff_eval import CocStuffEvaluator

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
    training_loss = 0 # Storing total loss
    
    # For each batch
    for batch, (images, targets) in enumerate(progress_bar(data_loader, parent=master_progress_bar)):
        
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

        master_progress_bar.child.comment = "Train loss: %.2f %s" % (training_loss/(batch + 1), log_loss(loss_dict_reduced))
        
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Return training loss
    return training_loss/ len(data_loader)


def evaluate(model: torch.nn.Module, 
             data_loader: torch.utils.data.DataLoader, 
             master_progressbar: fastprogress.master_bar,
             device: str="cpu"):
    
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Switch to model evaluation mode
    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocStuffEvaluator(coco)
    with torch.to_grad(): # Turn off gradient
        
        # For each batch
        for batch, (images, targets) in enumerate(progress_bar(data_loader, parent=master_progress_bar)):
            # Move images and targets to device
            images = list(image.to(device) for image in images)
            
            # Predict outputs
            outputs = model(image)
            
            # Move to CPU for evaluation
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
    
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
    
            
        