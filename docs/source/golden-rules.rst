Golden rules for scaling
========================

So, you want to scale your training, huh? The good news first: it's not too difficult. It boils down to a few simple principles and some basic linear algebra. The bad news? It requires unlearning a few concepts you may have been taught in lectures. For example, consider the following principle:

	Initialize the weights so that all activations have unit variance at initialization.

	-- Deep Learning 101

This turns out to be bad for scaling. Why? Because the network internals can behave quite differently at initialization compared to after a few steps of training. A good way to understand this point is to consider a simple linear layer.

The linear layer
^^^^^^^^^^^^^^^^^

Consider a linear layer with Gaussian initialization and standard deviation ``sigma``:

.. code:: python

    class Linear:

        def __init__(self, fan_out:int, fan_in:int, sigma:float):
            self.weight = sigma * torch.randn(fan_out, fan_in)

        def forward(self, x):
            return torch.matmul(self.weight, x)

The properties of this layer are most subtle when the layer conducts a large reduction in dimension---i.e. when ``fan_in`` is much larger than ``fan_out``. This might happen in the final layer of a classifier, for example. In fact, let's study the case where we are scaling up ``fan_in`` while holding ``fan_out`` fixed. 

An important fact about a matrix :python:`self.weight` with ``fan_in`` much larger than ``fan_out`` is that the null space is huge, meaning that most of the input space is mapped to zero. The dimension of the null space is at least ``fan_out - fan_in``. At initialization, most of a fixed input ``x`` will lie in this nullspace. This means that to get the output of :python:`self.forward` to have unit variance at initialization, you need to pick a huge initialization scale ``sigma`` in order to scale up the part of ``x`` that does not lie in the null space. But after a few steps of training, the situation changes. Gradient descent will cause the input ``x`` to align with the non-null space of ``self.weight``. This means that the ``sigma`` you chose to control the activations at initialization is now far too large in hindsight, and the activations will blow up! This problem only gets worse with increasing ``fan_in``.

The solution to this problem is simple: don't choose ``sigma`` to control variance at initialization! Instead, choose ``sigma`` under the assumption that inputs fall in the non-null space. Even if the activations are too small at initialization, this is fine as they will quickly "warm up" after a few steps of training. And finally we will show in `Fixing width scaling`_ that, if we switch from Gaussian intialization to orthogonal intialization, then choosing the right ``sigma`` becomes trivial.

Three golden rules
^^^^^^^^^^^^^^^^^^^

The example in the previous section illustrates a style of thinking that extends far beyond linear layers. Let's distill it into three key tenets, which we call the "golden rules" of scaling:

    .. rst-class:: starlist

        - Gradient descent causes inputs to align with the largest spectral components of tensors. So when initializing tensors, carefully set their largest spectral components.

        - The largest spectral components of gradient updates align with tensor inputs. So it is important to normalize gradient updates to control the size of their largest spectral components.

        - All layers will align during training. Keep this in mind when designing the architecture.

It's worth expanding a little on what we mean by *alignment* here. When we say that an input ``x`` aligns with a weight matrix ``weight``, we mean that if we compute ``U, S, V = torch.linalg.svd(weight)``, then the input ``x`` will tend to have a larger dot product with the rows of ``V`` that correspond to larger diagonal entries of the singular value matrix ``S``.

What's the source of this alignment? Well consider the following picture:

.. plot:: figure/alignment.py

A gradient update :math:`\Delta W` to a tensor :math:`W` "sees" both the head of the network (through the layer inputs) and the tail of the network (through the backpropagated gradient). Applying the update will align the head with the tail [#outerproduct]_. And this kind of alignment happens at all layers at every iteration!

The remainder of this section will show how to apply the golden rules to do width scaling, depth scaling, and key-query dot product scaling. This should already be enough for you to get started scaling a GPT.


Fixing width scaling
^^^^^^^^^^^^^^^^^^^^^

First, let's do width scaling in a linear layer. When the network has trained for a few steps to reach its fully aligned state, we want the input and output activations to fall roughly in the interval [-1, 1]. Applying the first golden rule, this tells us that the top singular values of the linear layer need to be around :python:`math.sqrt(fan_out / fan_in)` in magnitude. Intuitively, this is a "conversion factor" that turns a ``fan_in``-dimensional vector with order one entries to a ``fan_out``-dimensional vector with order one entries.

In fact we can be a little more clever here and reparameterize the linear layer as follows:

.. code:: python

    class ReparameterizedLinear:

        def __init__(self, fan_out:int, fan_in:int):
            self.scale = math.sqrt(fan_out / fan_in)
            self.weight = torch.empty(fan_out, fan_in)
            torch.nn.init.orthogonal_(self.weight)

        def forward(self, x):
            return self.scale * torch.matmul(self.weight, x)

By including the dimensional conversion factor :python:`self.scale = math.sqrt(fan_out / fan_in)` in the forward function, we can now initialize :python:`self.weight` to have order one singular values. Orthogonal init is a simple and interpretable way to do this---it makes all the singular values exactly one! In our experiments, we have found this parameterization and initialization to be performant without needing any hyperparameters. We use this style of reparameterized forward function in all Modula atomics.

As for training, we want gradient updates to be as large as possible without blowing up the outputs. By the second golden rule, this implies that we want the gradient updates to have singular values that are again order one. A simple way to achieve this is to just directly spectrally normalize [#spectralnorm]_ the updates:

.. code:: python

    self.weight -= learning_rate * self.weight.grad / spectral_norm(self.weight.grad)

In practice, you may want to replace :python:`self.weight.grad` with some kind of momentum or Adam expression. And the learning rate can optionally decay through the course of training.

Fixing depth scaling
^^^^^^^^^^^^^^^^^^^^^

For depth scaling, we will look at scaling the number of blocks in a residual network [#mlp]_ of the form:

.. code:: python

    def resnet(x:torch.Tensor, layer_list:list, block_multiplier:float):

        for layer in layer_list:

            x += block_multiplier * layer(x)

        return x

We call this a residual network because at each iteration of the :python:`for` loop, a small "residue" is added to the input, which takes the form of a neural network ``layer`` applied to the output from the previous iteration. The ``block_multiplier`` can be used to ensure that the residue is small, allowing us to make the residual network very, very deep. The main questions are:

- What kind of layers are we allowed to include in the ``layer_list``?
- What value should we choose for the ``block_multiplier``?

The third golden rule makes answering these questions easy. If we design each ``layer`` in ``layer_list`` in accordance with the golden rules [#modula]_, then we should set :python:`block_multiplier = 1 / len(layer_list)`. This is because each layer will add a contribution to the output, and there are :python:`len(layer_list)` layers in total. Since we are assuming that all layers align by the third golden rule, we should divide each contribution by :python:`len(layer_list)` in order to ensure that the output does not blow up.

This is similar to an idea you may have seen in math that :math:`(1+\frac{1}{L})^L < \mathrm{e}` for any :math:`L>0`. Even though the product may involve a large number :math:`L` of terms, the residue :math:`1/L` is small enough to prevent the product blowing up. Linking the analogy back to neural nets, :math:`L` plays the role of :python:`len(layer_list)`.

Fixing key-query dot product scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Warning
   :class: warning

   This section is still under construction.

.. [#outerproduct] The mathematical analogue of this intuitive argument is to say that the gradient of a matrix is an outer product of the layer input with the gradient of the loss with respect to the layer output.

.. [#spectralnorm] The spectral norm of a matrix is the largest singular value. The largest singular value of :python:`matrix / spectral_norm(matrix)` is always one, so long as :python:`matrix != 0`.

.. [#mlp] We study residual networks over MLPs because MLPs seem to just work bad beyond depth 10 or so. In the Modula paper, we show that the type of residual networks we propose are in fact "smooth" even in the limit of infinitely many blocks. The same property does not hold for MLPs to the best of our knowledge.

.. [#modula] The fact that this smells like a recursive statement was part of the inspiration for Modula.