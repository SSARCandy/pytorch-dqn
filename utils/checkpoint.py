import os
import torch

OUTPUT_DIR = 'checkpoints'

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(OUTPUT_DIR, filename))
    print("Checkpoint %s" % filename)


def load_checkpoint(filename):
    cp = torch.load(filename)

    del cp['target_state_dict']['fc5.bias']
    del cp['target_state_dict']['fc5.weight']
    del cp['q_state_dict']['fc5.bias']
    del cp['q_state_dict']['fc5.weight']

    return cp

def drop_unmatch_dict(dict1, dict2):
    filtered_dict = {k: v for k, v in dict2.items() if k in dict1}
    return filtered_dict
