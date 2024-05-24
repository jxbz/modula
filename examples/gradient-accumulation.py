import math

from torch import no_grad
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modula.atom import Linear
from modula.bond import ReLU, Flatten

micro_batch_size = 32
accumulate_steps = 5
log_steps = 10

# get MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset1, batch_size=micro_batch_size, num_workers=1, pin_memory=True, shuffle=True)

# create network
mlp = Linear(10,10000) @ ReLU() @ Linear(10000, 784) @ Flatten()
weights = mlp.initialize(device="cpu")

# train network
for step, (data, target) in enumerate(train_loader):
    target = one_hot(target, num_classes=10) * math.sqrt(10)
    output = mlp(data, weights)

    loss = (target - output).square().mean()
    loss.backward()
    acc = (target.argmax(dim=1) == output.argmax(dim=1)).sum() / micro_batch_size

    if step % accumulate_steps == 0:
        with no_grad():
            mlp.normalize(grad := weights.grad())
            weights -= 0.1 * grad
            weights.zero_grad()

            mlp.regularize(weights, strength = 0.01)

    if step == 0: print("step", "\t", "train acc")
    if step % log_steps == 0: print(step, '\t', acc.item())
