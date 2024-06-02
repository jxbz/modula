Golden rules for scaling
========================

So, you want to scale your training, huh? The good news first: it's not too difficult. It boils down to a few simple principles and some basic linear algebra. The bad news? It requires unlearning a few concepts you may have been taught in lectures. For example, consider the following principle:

	Initialize the weights so that all activations have unit variance at initialization.

	-- Deep Learning 101

This turns out to be bad for scaling. Why? Because the network internals can behave quite differently at initialization compared to after a few steps of training. A good way to understand this point is to consider a simple linear layer.

The linear layer
^^^^^^^^^^^^^^^^^

Consider a simple linear layer with initialization scale ``sigma``:

.. code:: python

    class Linear:

        def __init__(fan_out:int, fan_in:int, sigma:float):
            self.weight = sigma * torch.randn(fan_out, fan_in)

        def forward(x):
            return torch.matmul(self.weight, x)

Suppose that we are dealing with the output layer of a classifier, and we want to scale up ``fan_in`` while holding ``fan_out`` fixed. Then an important fact about :python:`self.weight` is that it has a huge null space of dimension ``fan_out - fan_in``. (The null space is the set of inputs that get mapped to zero). At initialization, most of an input ``x`` will overlap with this nullspace. This means that to get the output of :python:`self.forward` to have unit variance at initialization, you need to pick a huge initialization scale ``sigma``. But after a few steps of training, the situation changes. Gradient descent causes the input ``x`` to align with the non-null space of ``self.weight``. The ``sigma`` you chose to control the activations at initialization is now far too large in hindsight!

Two golden rules
^^^^^^^^^^^^^^^^^

The example in the previous section illustrates a style of thinking that extends far beyond linear layers. Let's distill it into two key principles, which we call the "golden rules" of scaling.

    🌟 Gradient descent causes layer inputs to align with the largest singular value components of weight matrices. Therefore, design the architecture and initialize the weight matrices under the assumption that inputs will align with the largest singular value components of all matrices.

    🌟 The largest singular value components of gradient updates align with layer inputs. Therefore, normalize gradient updates under the assumption that their largest singular value components will align with inputs at all layers.

It's worth expanding a little on what we mean by *alignment* here. When we say that an input ``x`` aligns with a weight matrix ``weight``, we mean that if we compute ``U, S, V = torch.linalg.svd(weight)``, then the input ``x`` will tend to have a larger dot product with the rows of ``V`` that correspond to larger diagonal entries of the singular value matrix ``S``.

Applying the rules
^^^^^^^^^^^^^^^^^^^

Now, let's apply the golden rules to understand how to do width scaling, depth scaling, and key-query dot product scaling. This should already be enough for you to get started scaling a GPT.
