
import torch

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print("Checkpoint saved %d" % state["timestep"])


def load_checkpoint(filename='checkpoint.pth.tar'):
    cp = torch.load(filename)
    return cp
