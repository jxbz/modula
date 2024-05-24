from torch import randn, no_grad
from modula.atom import Linear
from modula.bond import ReLU

data, target = randn(1000), randn(10)

mlp = Linear(10,10000) @ ReLU() @ Linear(10000, 1000)
weights = mlp.initialize(device="cpu")

for step in range(steps:=20):
    output = mlp(data, weights)

    loss = (target - output).square().mean()
    loss.backward()

    with no_grad():
        mlp.normalize(grad := weights.grad())
        weights -= 0.1 * grad
        weights.zero_grad()
    
        mlp.regularize(weights, strength = 0.01)

    if step == 0: print("step", "\t", "train loss")
    print(step, '\t', loss.item())
