import sys
import os
import signal
import math
import torch
import numpy
import argparse
import pickle

from tqdm import trange
from data.dataset import getIterator
from modula.compound import *

from misc import check_bfloat16_support, Scheduler

architectures = ['resmlp', 'rescnn', 'gpt']
datasets      = ['cifar10', 'shakespeare', 'openwebtext', 'tinystories']
losses        = ['mse', 'xent']
dtype         = ['bfloat16', 'float32', 'auto']
scheduler     = ["linear", "cosine", "none"]

parser = argparse.ArgumentParser()

# system
parser.add_argument('--cpu',            action='store_true'      )
parser.add_argument('--distribute',     action='store_true'      )
parser.add_argument('--log_dir',        type=str,   default='logs/temp')
parser.add_argument('--log_interval',   type=int,   default=100  )
parser.add_argument('--seed',           type=int,   default=0    )
parser.add_argument('--batch_size',     type=int,   default=128  )
parser.add_argument('--train_steps',    type=int,   default=1000 )
parser.add_argument('--test_steps',     type=int,   default=100  )
parser.add_argument('--dataset',        type=str,   default='cifar10',  choices=datasets)
parser.add_argument('--dtype',          type=str,   default='float32',  choices=dtype)

# architecture
parser.add_argument('--arch',           type=str,   default='resmlp',   choices=architectures)
parser.add_argument('--depth',          type=int,   default=6    )
parser.add_argument('--block_depth',    type=int,   default=2    )
parser.add_argument('--width',          type=int,   default=384  )
parser.add_argument('--context',        type=int,   default=256  )
parser.add_argument('--num_heads',      type=int,   default=8    )

# training
parser.add_argument('--normalize',      type=int,   default=1     ,     choices=[0,1])
parser.add_argument('--loss',           type=str,   default='xent',     choices=losses)
parser.add_argument('--lr',             type=float, default=0.5  )
parser.add_argument('--beta1',          type=float, default=0.9  )
parser.add_argument('--beta2',          type=float, default=0.99 )
parser.add_argument('--wd',             type=float, default=0.01 )
parser.add_argument('--scheduler',      type=str,   default="linear", choices=scheduler)
parser.add_argument('--min_lr_factor',  type=float, default=0.0  )


def set_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)

def evalute(output, data, target):

    if args.arch == "gpt":
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

    acc = (output.argmax(dim=1) == target).sum() / target.numel()

    if args.loss == 'mse':
        onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
        error = (output - onehot * math.sqrt(output.shape[1])).square().mean(dim=1)

    elif args.loss == 'xent':
        error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)

    loss = error.mean()

    return loss, acc


if __name__ == '__main__':

    args = parser.parse_args()

    if args.distribute:
        torch.distributed.init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        pickle.dump(vars(args), open( os.path.join(args.log_dir, 'args.pickle'), "wb" ) )
        for arg in vars(args):
            print("{: <20} {: <20}".format(arg, getattr(args, arg)))

    set_seed(args.seed + rank)
    getBatch, input_dim, output_dim = getIterator(  dataset = args.dataset,
                                                    batch_size = args.batch_size,
                                                    context = args.context  )

    def cleanup(sig=None, frame=None):
        global getBatch
        del getBatch
        print("Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    if args.arch == "resmlp":
        net = ResMLP(args.width, args.depth, args.block_depth, math.prod(input_dim), output_dim)

    elif args.arch == "rescnn":
        net = ResCNN(args.width, args.depth, args.block_depth, input_dim[0], output_dim)

    elif args.arch == "gpt":
        net = GPT(  vocab_size = input_dim,
                    context = args.context,
                    num_heads = args.num_heads,
                    d_embed = args.width,
                    d_query = args.width // args.num_heads,
                    d_value = args.width // args.num_heads,
                    num_blocks = args.depth )

    if rank == 0: print(net)

    if args.dtype == 'auto':
        dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    device = "cpu" if args.cpu else rank

    set_seed(args.seed)
    weights = net.initialize(device=device, dtype=dtype)
    set_seed(args.seed + rank)

    with torch.no_grad():
        mom1 = 0 * weights
        if args.beta2 >= 0:
            mom2 = 0 * weights

    results = {"train_loss":[], "test_loss":[], "train_acc":[], "test_acc":[]}

    for step in (pbar := trange(args.train_steps + 1, file=sys.stdout, disable=(rank != 0))):

        if step % args.log_interval == 0:
            test_loss = test_acc = 0
            for _ in range(args.test_steps):
                data, target = getBatch(train = False)
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.no_grad(): loss, acc = evalute(net.forward(data, weights), data, target)

                test_loss += loss
                test_acc += acc

            results["test_loss"].append(test_loss.item() / args.test_steps)
            results["test_acc"].append(test_acc.item() / args.test_steps)

        data, target = getBatch(train = True)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        train_loss, train_acc = evalute(net.forward(data, weights), data, target)
        train_loss.backward()

        with torch.no_grad():
            grad = weights.grad()
            if args.distribute: grad.all_reduce()

            if args.beta2 >= 0:
                mom1 += (1-args.beta1)**(step/(step+1)) * (grad    - mom1)
                mom2 += (1-args.beta2)**(step/(step+1)) * (grad**2 - mom2)
                update = mom1 / mom2 ** 0.5
            else:
                mom1 += (1-args.beta1)**(step/(step+1)) * (grad    - mom1)
                update = mom1 * 1.0

            update.zero_nans()

            schedule = Scheduler.get_lr(args.scheduler, step, args.train_steps, args.min_lr_factor)

            if args.normalize:
                net.normalize(update, target_norm = args.lr * schedule)
            else:
                update *= args.lr * schedule
            weights -= update
            net.regularize(weights, strength = args.lr * schedule * args.wd)
            weights.zero_grad()

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())

        if rank == 0 and step % args.log_interval == 0:
            pickle.dump(results, open( os.path.join(args.log_dir, 'results.pickle'), "wb" ) )
            pbar.set_description(f"train: {numpy.mean(results['train_acc'][-100:]):.4f} // test: {results['test_acc'][-1]:.4f}")

            if step > 0 and math.isnan(train_loss): cleanup()

    cleanup()
