Bad scaling
============

At the simplest level, neural networks are trained by iterating the following operation:

.. code:: python

   weights -= learning_rate * gradient

where :python:`learning_rate` is a :python:`float` and :python:`gradient` is the gradient of the loss function with respect to the :python:`weights` of the network. Of course, in practice, we may want to use additional tricks such as momentum, but let's ignore details like that for now.

Unfortunately, this simple "gradient descent" operation does not scale well if we scale up the network architecture. What does this mean? Suppose that, before training, we "grow" the network by increasing its *width* (the number of neurons in a layer) or its *depth* (the number of layers):

.. plot:: figure/nn.py

In practice, we might like to grow other dimensions such as the number of residual blocks in a transformer, but let's stick with this simplified picture for now. 

Under these scaling operations, gradient descent training can break in two main ways. The first problem is that the optimal learning rate can *drift* as we scale certain dimensions. This is a problem because it means we need to re-tune the learning rate as we scale things up---which is expensive and time-consuming. The second problem is that sometimes performance can actually get worse as we grow the network, even if the optimal learning rate remains stable. This is a problem because we grew the network hoping to make performance better, not worse!

.. plot:: figure/sweeps.py

   These cartoons illustrate typical bad scaling behaviours. On the left, the optimal learning rate drifts with increasing width. On the right, performance deteriorates with increasing depth.

The good news is that we have developed machinery that largely solves these scaling woes. It turns out that the problem is solved by defining a simple weight initializer along with a special :python:`normalize` function which acts on gradients, leading to a new "normalized" gradient descent algorithm:

.. code:: python

   weights -= learning_rate * normalize(gradient)

This initialization and gradient normalization removes drift in the optimal learning rate, and causes performance to improve with increasing scale. Modula automatically infers the necessary initialize and normalize functions from the architecture of the network. So the user can focus on writing their neural network architecture while Modula will handle properly normalizing the training. 

These docs are intended to explain how Modula works and also introduce the Modula API. In case you don't care about Modula or automatic gradient normalization, the next section will explain how you can normalize training manually in a different framework like `PyTorch <https://pytorch.org>`_ or `JAX <https://github.com/google/jax>`_.