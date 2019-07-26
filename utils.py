import os
import torch
import torch.distributed as dist

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def log_loss(input_dict: dict):
    log = ""
    for k, v in input_dict.items():
        log += "%s: %.2f\t" % (k.replace("loss_", ""), v.item())
    return log


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict: dict, average: bool=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.
    
    Parameters
    ----------
    input_dict: dict
        all the values will be reduced
    average: bool
        whether to do average or sum
        
    Returns
    -------
    a dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def collate_fn(batch):
    return tuple(zip(*batch))

import numpy as np
def standardlize_masks(masks):
    no_masks = masks.size()[0]
    new_masks = masks.cpu().numpy()
    if no_masks == 1:
        return new_masks
    for i in range(1,no_masks):
        for j in range(i-1,-1,-1):
            current_mask = new_masks[i,:,:]
            current_mask -= new_masks[j,:,:]
            current_mask[current_mask < 0] = 0
            current_mask[current_mask > 1] = 0
            new_masks[i,:,:] = current_mask
    return new_masks