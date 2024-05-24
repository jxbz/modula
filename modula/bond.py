import math
import torch

from modula.abstract import Module
from modula.vector import Vector


class Bond(Module):
    """A module with no weights."""
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.length = 0
        self.initialize = lambda device, dtype : Vector()
        self.normalize  = lambda w, target_norm : None
        self.regularize = lambda w, strength : None


class Identity(Bond):
    """Identity module."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x


class Flatten(Bond):
    """Flatten all non-batch dimensions."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.flatten(start_dim=1)


class AddHeads(Bond):
    """Reshapes an input to have heads.

    Input shape: batch_size, sequence_length, embed_dim
    Output shape: batch_size, num_heads, sequence_length, head_size

    Adapted from Karpathy's nanoGPT.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.sensitivity = 1
        self.num_heads = num_heads

    def forward(self, x, w):
        B, T, C = x.size()
        return x.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)


class RemoveHeads(Bond):
    """Inverse of AddHeads."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1

    def forward(self, x, w):
        B, nh, T, hs = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nh*hs)


class Enumerate(Bond):
    """Replace each column with its column index. Used to make position embeddings."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.arange(0, x.size()[1], dtype=torch.long, device=x.device)


class Abs(Bond):
    """Absolute value nonlinearity."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.abs(x)


class ReLU(Bond):
    """ReLU nonlinearity."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: torch.nn.functional.relu(x)


def ScaledReLU():
    """ReLU scaled to have sensitivity one."""
    return math.sqrt(2) * ReLU()


class GELU(Bond):
    """GELU nonlinearity."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: torch.nn.functional.gelu(x)


def ScaledGELU():
    """GELU scaled to have sensitivity 1."""
    return math.sqrt(2) * GELU()


class MeanSubtract(Bond):
    """Mean subtraction."""
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x - x.mean(dim=dim, keepdim=True)


class RMSDivide(Bond):
    """Normalize to have unit RMS norm."""
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x / x.square().mean(dim=dim, keepdim=True).sqrt()


def LayerNorm(dim=-1):
    """Mean subtraction followed by RMS normalization."""
    return RMSDivide(dim) @ MeanSubtract(dim)


class Mean(Bond):
    """Take the mean over a specified dimension."""
    def __init__(self, dim):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.mean(dim=dim)


class AvgPool(Bond):
    """Average pooling that adapts to different input sizes."""
    def __init__(self, output_size = (1,1)):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.nn.functional.adaptive_avg_pool2d(x, output_size)


class FunctionalAttention(Bond):
    """The part of attention that doesn't involve weights."""

    def __init__(self, causal):
        super().__init__()
        self.sensitivity = 1
        self.causal = causal

    def forward(self, x, w):
        q, k, v = x

        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=1/q.shape[-1])