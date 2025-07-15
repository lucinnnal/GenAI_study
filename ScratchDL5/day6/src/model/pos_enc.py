import torch
import torch.nn as nn

# Convert integer type timestep t to D dim vector
def _pos_encoding(t, output_dim, device='cpu'):
    D = output_dim
    v = torch.zeros(D, device=device)

    index = torch.arange(0, D, device=device)
    div = 10000 ** (index / D)

    v[0::2] = torch.sin(t / div[0::2])
    v[1::2] = torch.cos(t / div[1::2])

    return v

# Positional Encoding For Multile Timesteps which are integers
def pos_encoding(ts, output_dim, device='cpu'):
    batch_size = len(ts) # number of t for positional encoding
    v = torch.zeros(batch_size, output_dim, device=device)

    for i in range(batch_size): v[i] = _pos_encoding(ts[i], output_dim, device)

    return v