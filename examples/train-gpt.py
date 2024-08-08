import time
import torch
import numpy as np

# Karpathy's smallest GPT config

vocab_size = 65
context = 64
num_heads = 4
d_embed = 128
d_query = 32
d_value = 32
num_blocks = 4

# Llama-7b-like values, excluding the vocabulary size.
vocab_size = 256
context = 1024
num_heads = 32
d_embed = 4096
d_query = 128
d_value = 128
num_blocks = 4

GPU_16BIT_FLOPS = {
    "h100-sxm": 1.979e15 / 2,
    "h100-pcie": 1.513e15 / 2,
    "a100": 312e12,
    "v100-sxm": 125e12,
    "6000A": 364.25e12,
    "4090": 165.2 * 10**12,
    "3090": 71 * 10**12,
    "t4": 65e12,
}
def xf_layer_fwd_flops(slen: int, bs: int=1, causal=True) -> int:
    p_mlp = d_embed * 4 * d_embed * 2
    f_mlp = p_mlp * 2 * slen

    assert d_query == d_value, "Dq != Dv not implemented"
    p_att = 4 * d_embed * d_embed
    f_att = p_att * 2 * slen
    f_sdpa = 4 * slen * slen * d_embed // (2 if causal else 1) # approximation

    return (f_mlp + f_att + f_sdpa) * bs

def gpt_train_flops(slen: int, bs: int, causal=True) -> int:
    # lmhead layer:
    flops = 6 * slen * bs * d_embed * vocab_size
    # assume no activation checkpointing
    flops += num_blocks * xf_layer_fwd_flops(slen, bs, causal) * 3
    return flops

class SpeedLogger:
    def __init__(self, ideal_flops_per_sec: float):
        self.tps = []
        self.mfu = []
        self.fps = ideal_flops_per_sec

    def add(self, slen: int, bs: int, duration: float) -> tuple[float,float]:
        flops = gpt_train_flops(slen, bs)
        self.tps.append(slen*bs / duration)
        self.mfu.append(flops / duration / self.fps)
        return self.tps[-1], self.mfu[-1]

    def ave(self):
        return sum(self.tps) / len(self.tps), sum(self.mfu) / len(self.mfu)

# training hparams

init_lr = 0.5
wd = 0.01
batch_size = 2 # 12
steps = 2001
eval_steps = 100
log_interval = 10 # 200

# let's start by defining our GPT architecture
# (we could instead just import GPT from modula.compound)

from modula.atom import *
from modula.bond import *

def Attention(num_heads, d_embed, d_query, d_value, context, causal):
    """Multi-head attention."""
    Q = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = AddHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, d_value * num_heads) @ RemoveHeads()

    return W @ FunctionalAttention(causal) * 1/3 @ (Q, K, V)

def GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5):
    """GPT."""
    token_embedding = Embedding(vocab_size, d_embed)
    position_embedding = Embedding(context, d_embed) @ Enumerate()
    initial = 1/2 * token_embedding + 1/2 * position_embedding
    initial.tare()

    attention = Attention(num_heads, d_embed, d_query, d_value, context, causal=True) @ LayerNorm()
    mlp = Linear(d_embed, 4*d_embed) @ ScaledGELU() @ Linear(4*d_embed, d_embed) @ LayerNorm()
    attention_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * attention
    mlp_block       = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ attention_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    final = Linear(vocab_size, d_embed) @ LayerNorm()

    return final @ blocks @ initial

# now let's set up some data loading utils

class RandomSampler(torch.utils.data.Sampler):

    def __init__(self, data, batch_size):
        self.length = len(data)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield np.random.randint(self.length, size=self.batch_size)

class SimpleLLMDataset(torch.utils.data.Dataset):

    def __init__(self, data, context):
        self.data = data
        self.context = context

    def __getitem__(self, index):
        return torch.tensor(self.data[index  :index+self.context  ].astype(np.int64)), \
               torch.tensor(self.data[index+1:index+self.context+1].astype(np.int64))

    def __len__(self):
        return len(self.data) - self.context - 1

# now let's start doing stuff


@torch.cuda.amp.autocast(dtype=torch.bfloat16)
def train(device, ideal_flops_per_sec):
    # load the data

    trainset = SimpleLLMDataset(np.memmap("examples/data/shakespeare/train.bin", dtype=np.uint16, mode='r'), context)
    testset  = SimpleLLMDataset(np.memmap("examples/data/shakespeare/val.bin",   dtype=np.uint16, mode='r'), context)

    train_sampler = RandomSampler(trainset, batch_size)
    test_sampler  = RandomSampler(testset,  batch_size)

    train_loader = torch.utils.data.DataLoader( trainset, num_workers=1, pin_memory=True, batch_sampler=train_sampler)
    test_loader  = torch.utils.data.DataLoader( testset,  num_workers=1, pin_memory=True, batch_sampler=test_sampler)

    train_iterator = iter(train_loader)
    test_iterator  = iter(test_loader)

    def getBatch(train: bool) -> list:
        res = next(train_iterator if train else test_iterator)
        return [t.to(device=device) for t in res]

    # load the model

    gpt = GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks)
    weights = gpt.initialize(device=device)
    gpt.forward = torch.compile(gpt.forward)
    # gpt.normalize = torch.compile(gpt.normalize)
    # gpt.regularize = torch.compile(gpt.regularize)
    # init_lr_t = torch.tensor(init_lr, device=device)

    # initialize the Adam state

    beta1 = 0.9
    beta2 = 0.99

    with torch.no_grad():
        mom1 = 0 * weights
        mom2 = 0 * weights

    # train the model

    speed_logger = SpeedLogger(ideal_flops_per_sec)

    for step in range(steps):

        if step % log_interval == 0:
            test_loss = test_acc = 0
            for eval_step in range(eval_steps):
                data, target = getBatch(train = False)
                output = gpt.forward(data, weights)
                output = output.view(-1, output.size(-1))
                target = target.view(-1)

                with torch.no_grad():
                    test_acc += (output.argmax(dim=1) == target).sum() / target.numel()
                    error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)
                    test_loss += error.mean()
            test_loss /= eval_steps
            test_acc /= eval_steps

        t0 = time.time()
        data, target = getBatch(train = True)
        output = gpt.forward(data, weights)
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

        train_acc = (output.argmax(dim=1) == target).sum() / target.numel()
        error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)
        train_loss = error.mean()

        train_loss.backward()

        with torch.no_grad():
            grad = weights.grad()

            # adam logic
            mom1 += (1-beta1)**(step/(step+1)) * (grad    - mom1)
            mom2 += (1-beta2)**(step/(step+1)) * (grad**2 - mom2)
            update = mom1 / mom2 ** 0.5
            update.zero_nans()

            schedule = 1 - step / steps

            # modular normalization and weight update
            gpt.normalize(update, target_norm = init_lr * schedule)
            weights -= update
            gpt.regularize(weights, strength = init_lr * schedule * wd)
            weights.zero_grad()

        # avoid first compile && first recompile
        if step > 1:
            speed_logger.add(*data.shape, time.time() - t0)

        if step > 1 and step % log_interval == 0:
            tps, mfu = speed_logger.ave()
            print(
                "step:", step,
                "\t train loss:", "%.2f" % train_loss.item(),
                "\t test loss:",  "%.2f" % test_loss.item(),
                f"\t tokens/gpu/sec: {tps:.2f}",
                f"\t MFU: {mfu*100:.2f}%",
            )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    train('cuda' if args.cuda else 'cpu', GPU_16BIT_FLOPS['3090'])
