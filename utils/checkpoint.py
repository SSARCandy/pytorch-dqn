import os
import torch

OUTPUT_DIR = 'checkpoints'

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(OUTPUT_DIR, filename))
    print("Checkpoint %s" % filename)


def load_checkpoint(filename, Q_net, target_Q_net):
    cp = torch.load(filename)
    Q_dict = Q_net.state_dict()
    target_Q_dict = target_Q_net.state_dict()
    cp['q_state_dict'] = drop_unmatch_dict(Q_dict, cp['q_state_dict'])
    cp['target_state_dict'] = drop_unmatch_dict(target_Q_dict, cp['target_state_dict'])
    return cp

def drop_unmatch_dict(dict1, dict2):
    filtered_dict = {k: v for k, v in dict2.items() if k in dict1}
    return filtered_dict
