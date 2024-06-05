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

The properties of this layer are most subtle when the layer conducts a large reduction in dimension---i.e. when ``fan_out`` is much smaller than ``fan_in``. This might happen in the final layer of a classifier, for example. In fact, let's study the case where we are scaling up ``fan_in`` while holding ``fan_out`` fixed. 

An important fact about a matrix :python:`self.weight` with ``fan_in`` much larger than ``fan_out`` is that the null space is huge, meaning that most of the input space is mapped to zero. The dimension of the null space is at least ``fan_in - fan_out``. At initialization, most of a fixed input ``x`` will lie in this nullspace. This means that to get the output of :python:`self.forward` to have unit variance at initialization, you need to pick a huge initialization scale ``sigma`` in order to scale up the component of ``x`` that does not lie in the null space. But after a few steps of training, the situation changes. Gradient descent will cause the input ``x`` to align with the non-null space of ``self.weight``. This means that the ``sigma`` you chose to control the activations at initialization is now far too large in hindsight, and the activations will blow up! This problem only gets worse with increasing ``fan_in``.

The solution to this problem is simple: don't choose ``sigma`` to control variance at initialization! Instead, choose ``sigma`` under the assumption that inputs fall in the non-null space. Even if this makes the activations too small at initialization, this is fine as they will quickly "warm up" after a few steps of training. And for a nice bonus, we will show in the section on `width scaling <#fixing-width-scaling>`_ that switching from Gaussian init to orthogonal init makes choosing the right ``sigma`` trivial.

Three golden rules
^^^^^^^^^^^^^^^^^^^

The example in the previous section illustrates a style of thinking that extends far beyond linear layers. Let's distill it into three key tenets, which we call the "golden rules" of scaling:

    .. rst-class:: starlist

        - Gradient descent causes inputs to align with the largest spectral components of tensors. So when initializing tensors, carefully set their largest spectral components.

        - The largest spectral components of gradient updates align with tensor inputs. So it is important to normalize gradient updates to control the size of their largest spectral components.

        - All layers will align during training. Keep this in mind when designing the architecture.

It's worth expanding a little on what we mean by *alignment* here. When we say that an input ``x`` aligns with a weight matrix ``weight``, we mean that if we compute ``U, S, V = torch.linalg.svd(weight)``, then the input ``x`` will tend to have a larger dot product with the rows of ``V`` that correspond to larger diagonal entries of the singular value matrix ``S``. When we say that layers align, we mean that the outputs of one layer will align with the next layer.

What's the source of this alignment? Consider making a gradient update to a tensor in the middle of a deep net. We call all the preceding layers the "head" of the network, and all the layers after the "tail":

.. plot:: figure/alignment.py

What's important is that the gradient update "knows about" both the head of the network (through the layer inputs) and the tail of the network (through the backpropagated gradient). Applying the update will align the head with the tail [#outerproduct]_. And this kind of alignment happens at all layers at every iteration!

The rest of this section will show how to apply the golden rules to do `width scaling <#fixing-width-scaling>`_, `depth scaling <#fixing-depth-scaling>`_, and `key-query dot product scaling <#fixing-key-query-dot-product-scaling>`_. This should already be enough to get started scaling a GPT.

Fixing width scaling
^^^^^^^^^^^^^^^^^^^^^

First, let's do width scaling in a linear layer. When the network has trained for a few steps to reach its fully aligned state, we want the input and output activations to fall roughly in the interval [-1, 1]. Equivalently, we want the inputs to have Euclidean length :python:`math.sqrt(fan_in)` and the outputs to have Euclidean length :python:`math.sqrt(fan_out)`. To achieve this, the first and second golden rules tell us that we need to control the top singular values of the initial weight matrix and the gradient updates. One can check that the right scaling is to set the singular values proportional to :python:`math.sqrt(fan_out / fan_in)`. Intuitively, the factor of :python:`math.sqrt(fan_out / fan_in)` means that the matrix operates as a "dimensional converter": it takes in vectors of length :python:`math.sqrt(fan_in)` and spits out vectors of length :python:`math.sqrt(fan_out)`.

In fact we can be a little more clever here and reparameterize the linear layer as follows:

.. code:: python

    class ReparameterizedLinear:

        def __init__(self, fan_out:int, fan_in:int):
            self.scale = math.sqrt(fan_out / fan_in)
            self.weight = torch.empty(fan_out, fan_in)
            torch.nn.init.orthogonal_(self.weight)

        def forward(self, x):
            return self.scale * torch.matmul(self.weight, x)

By including the conversion factor :python:`self.scale = math.sqrt(fan_out / fan_in)` in the forward function, the correct scaling is to make the largest singular values of both :python:`self.weight` and the weight updates order one. Easy, right? For the initialization, we can just use orthogonal init, which sets all the singular values to exactly one. In our experiments, we have found orthogonal init to be a performant, hyperparameter-free initializer. As for weight updates, we can just spectrally normalize them [#spectralnorm]_:

.. code:: python

    self.weight -= learning_rate * self.weight.grad / spectral_norm(self.weight.grad)

In practice, you may want to replace :python:`self.weight.grad` with some kind of momentum or Adam expression. And the learning rate can optionally decay through the course of training.

Fixing depth scaling
^^^^^^^^^^^^^^^^^^^^^

For depth scaling, we will look at scaling the number of blocks in a residual network [#mlp]_ of the form:

.. code:: python

    def resnet(x:torch.Tensor, residue_list:list, block_multiplier:float):

        for residue in residue_list:

            x += block_multiplier * residue(x)

        return x

We call this a residual network because at each iteration of the :python:`for` loop, a :python:`residue` is added to the input, which takes the form of a sub-network applied to the output from the previous step of the loop. The ``block_multiplier`` can be used to ensure that the residual contribution is small, allowing us to make the residual network very, very deep without its output blowing up. The main questions are:

- What kind of functions are we allowed to include in the ``residue_list``?
- What value should we choose for the ``block_multiplier``?

The third golden rule makes answering these questions easy. We should set :python:`block_multiplier = 1 / len(residue_list)`. This is because each residue adds one contribution to the output, and there are :python:`len(residue_list)` residues in total. The sum of :python:`len(residue_list)` aligned residues needs to be divided by :python:`len(residue_list)` in order to not blow up. This is similar to an idea you may have seen in math that :math:`(1+\frac{1}{L})^L < \mathrm{e}` for any :math:`L>0`. Even though the product may involve a large number :math:`L` of terms, the residue :math:`1/L` is small enough to prevent the product blowing up. Linking the analogy back to neural nets, :math:`L` plays the role of :python:`len(residue_list)`.

Since the :python:`1/len(residue_list)` block multiplier prevents both the initialization and the updates to the residues from blowing up, we are safe to set each residue equal to any neural network of our choosing, so long as that network is individually initialized and updated in accordance with the golden rules [#recursive]_.

Fixing key-query dot product scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important operation in transformers is taking the dot product between key and query vectors. Conventionally this is done as follows:

.. code:: python

    lambda key, query : torch.dot(key, query) / math.sqrt(key.shape[0])

The factor of :python:`1 / math.sqrt(key.shape[0])` is included to prevent the dot product from blowing up at initialization, where we assume that ``key`` and ``query`` are uncorrelated random vectors. But by the golden rules, we should expect that the keys and queries become aligned with each other through the course of training. Therefore we should instead normalize the dot product as follows:

.. code:: python

    lambda key, query : torch.dot(key, query) / key.shape[0]

To spell this out more clearly, the dot product is the sum of a number :python:`key.shape[0]` of aligned quantities, so we should divide by :python:`key.shape[0]` to prevent the sum blowing up.

Wrapping up
^^^^^^^^^^^^

On this page, we introduced three "golden rules" for scaling and pointed out how they differ to some conventional wisdom about controlling activation variance at initialization. One of the points we hope to get across is that the logical reasoning associated with the golden rules is not only *more scalable* but also *simpler* than standard approaches based on controlling variance. You don't need to know anything about how random variables behave in order to get scaling right---you just need to know how objects add when they point in the same direction. Furthermore, the use of orthogonal initialization obviates the need to know anything about the spectral properties of Gaussian random matrices.

In the next section we will look at the history behind these ideas, and after that we will explain how Modula automates the application of the golden rules.


.. [#outerproduct] The mathematical analogue of this intuitive statement is to say that the gradient of a linear layer is an outer product of the layer input with the gradient of the loss with respect to the layer output.

.. [#spectralnorm] The spectral norm of a matrix is the largest singular value. The largest singular value of :python:`matrix / spectral_norm(matrix)` is always one, so long as :python:`matrix != 0`.

.. [#mlp] We study residual networks over MLPs because MLPs seem to just work bad beyond depth 10 or so. In the Modula paper, we show that the type of residual networks we propose are in fact "smooth" even in the limit of infinitely many blocks. The same property does not hold for MLPs to the best of our knowledge.

.. [#recursive] The recursive nature of this statement directly inspired the Modula framework.
