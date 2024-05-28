Welcome to the Modula docs!
============================

Modula is a deep learning framework designed for graceful scaling. The user defines a compound module (i.e. neural network) in Modula by arbitrarily composing atom and bond modules (e.g. linear layers and nonlinearities). Modula then automatically normalizes weight updates in the modular norm corresponding to this compound. This leads to automatic learning rate transfer across width, depth and possibly other architectural dimensions. Modula is built on top of `PyTorch <http://pytorch.org>`_.

Useful links:

* `Modula codebase <http://github.com/jxbz/modula>`_
* `arXiv paper <https://arxiv.org/abs/2405.14813>`_

.. toctree::
   :hidden:
   :maxdepth: 2

   quickstart

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation:

   theory/module
   theory/atom/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modula in Code:
