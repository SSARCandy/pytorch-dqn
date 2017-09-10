import os
import torch

OUTPUT_DIR = 'checkpoints'

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(OUTPUT_DIR, filename))
    print("Checkpoint %s" % filename)


def load_checkpoint(filename='checkpoint.pth.tar'):
    cp = torch.load(filename)
    return cp
