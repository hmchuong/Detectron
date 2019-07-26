"""Train the COCO Stuff dataset
"""
import os
import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.optim as optim

from fastprogress import master_bar
from tensorboardX import SummaryWriter

from dataset import COCOStuffDataset
from utils import collate_fn, init_distributed_mode
from model import get_mask_rcnn
from engine import train_one_epoch, evaluate

def main(args):
    init_distributed_mode(args)
    device = torch.device(args.device)
    writer = SummaryWriter(args.log_dir)
    
    train_dataset = COCOStuffDataset(image_dir=args.train_imagedir, annotation_dir=args.train_annodir)
    val_dataset = COCOStuffDataset(image_dir=args.val_imagedir, annotation_dir=args.val_annodir)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=1, num_workers=args.num_workers, collate_fn=collate_fn)
    
    model = get_mask_rcnn(182+1)
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    
    if device == "cuda":
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    
    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    mb_progressbar = master_bar(range(args.epochs))
    Path(args.log_dir).mkdir(exist_ok=True)
    for epoch in mb_progressbar:
        train_sampler.set_epoch(epoch)
        # Training
        mean_loss, loss_dict = train_one_epoch(model, optimizer, train_dataloader, mb_progressbar, device)
        writer.add_scalar('train_mean_loss', mean_loss, epoch)
        for k, v in loss_dict.items():
            writer.add_scalar(k, v, epoch)
        
        # Evaluating
        metrics = evaluate(model, val_dataloader, mb_progressbar, device)
        metric_names = ['mean_iou', 'fw_iou', 'mean_accuracy', 'pixel_accuracy', 'super_mean_iou', 'super_fw_iou', 'super_mean_accuracy', 'super_pixel_accuracy']
        for i, metric_name in enumerate(metric_names):
            writer.add_scalar(metric_name, metrics[i], epoch)
            
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.log_dir, "model-{}.pth".format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for COCO Stuff dataset"
    )
    parser.add_argument("--train-imagedir",
                        default="../cocostuff/dataset/images/train2017",
                        help="Image directory of training set")
    parser.add_argument("--train-annodir",
                        default="../cocostuff/dataset/annotations/train2017",
                        help="Annotation directory of training set")
    parser.add_argument("--val-imagedir",
                        default="../cocostuff/dataset/images/val2017",
                        help="Image directory of validation set")
    parser.add_argument("--val-annodir",
                        default="../cocostuff/dataset/annotations/val2017",
                        help="Annotation directory of validation set")
    parser.add_argument("--num-workers",
                        type=int,
                        default=34,
                        help="Number of CPU cores to process data")
    parser.add_argument("--batch-size",
                        type=int,
                        default=2,
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="Local rank")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    args = parser.parse_args()
    main(args)
