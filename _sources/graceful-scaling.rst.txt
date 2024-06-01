Graceful scaling
=================

.. admonition:: Warning
   :class: warning

   This page is still under construction.

At the simplest level, neural networks are trained by iterating the following operation:

.. code:: python

   weights -= learning_rate * gradient_of_loss

where :python:`learning_rate` is a :python:`float` and :python:`gradient_of_loss` is the gradient of the loss function with respect to its :python:`weights`. Of course, in practice, we may want to use additional tricks such as momentum, but let's ignore details like that for now.

Unfortunately, this simple "gradient descent" operation does not scale well if we scale up the network architecture. What does this mean? Consider "growing" the network by increasing the *width* (the number of neurons in a layer) or the *depth* (the number of layers):

``<insert figure>``

Then gradient descent training can break in two different ways:

``<insert figure>``

The picture on the left shows the optimal learning rate *drifting* as we scale up the width of the network. The picture on the right shows performance *getting worse* as we scale up the depth of the network.

It turns out that we can solve these problems by defining a special :python:`normalize` function which acts on gradients, leading to a new "normalized" gradient descent algorithm:

.. code:: python

   weights -= learning_rate * normalize(gradient_of_loss)

Modula automatically infers the required normalize function from the architecture of the network. So the user can focus on writing their neural network architecture while Modula will handle properly normalizing the training. These docs are intended to explain how Modula works and also introduce the Modula API.

In case you don't care about Modula and automatic normalization, the next section will explain how you can normalize training manually in a different framework like `PyTorch <https://pytorch.org>`_ or `JAX <https://github.com/google/jax>`_.