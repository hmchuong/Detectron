"""Train the COCO Stuff dataset
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from fastprogress import master_bar

from dataset import COCOStuffDataset
from utils import collate_fn
from model import get_mask_rcnn
from engine import train_one_epoch, evaluate

def main(args):
    
    train_dataset = COCOStuffDataset(image_dir=args.train_imagedir, annotation_dir=args.train_annodir)
    val_dataset = COCOStuffDataset(image_dir=args.val_imagedir, annotation_dir=args.val_annodir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    model = get_mask_rcnn(182+1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    mb_progressbar = master_bar(range(args.epochs))
    Path(args.log_dir).mkdir(exist_ok=True)
    for epoch in mb_progressbar:
        # Training
        loss = train_one_epoch(model, optimizer, train_dataloader, mb_progressbar, device)
        print("TRAIN")
        # Evaluating
        evalutator = evaluate(model, val_dataloader, mb_progressbar, device)
        print("DONE EVAL")
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.log_dir, "model-{}.pth".format(epoch)))
        
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for COCO Stuff dataset"
    )
    parser.add_argument("--train-imagedir",
                        default="samples/images",
                        help="Image directory of training set")
    parser.add_argument("--train-annodir",
                        default="samples/annotations",
                        help="Annotation directory of training set")
    parser.add_argument("--val-imagedir",
                        default="samples/images",
                        help="Image directory of validation set")
    parser.add_argument("--val-annodir",
                        default="samples/annotations",
                        help="Annotation directory of validation set")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of CPU cores to process data")
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        help="Batch size")
    parser.add_argument("--lr",
                        type=float,
                        default=0.005,
                        help="Learning rate")
    parser.add_argument("--lr_step",
                        type=int,
                        default=3,
                        help="Number of steps to reduce learning rate")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs")
    parser.add_argument("--log-dir",
                        type=str,
                        default="results",
                        help="Log directory")
    
    args = parser.parse_args()
    main(args)
