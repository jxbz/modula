Welcome to the Modula docs!
============================

Modula is a deep learning framework designed for graceful scaling. Neural networks written in Modula automatically transfer learning rate across scale. Modula can be installed by running: 

.. code-block:: console

   $ pip install modula

Purpose of the docs
^^^^^^^^^^^^^^^^^^^^

The purpose of these docs is three-fold: 

1. teach scaling through code;
2. introduce the Modula API;
3. explain how to extend Modula.

Navigating the docs
^^^^^^^^^^^^^^^^^^^^

If you don't care about Modula and just want to learn how to directly scale training in `PyTorch <https://pytorch.org>`_ or `JAX <https://github.com/google/jax>`_, then skip directly to the section on `golden rules for scaling <golden-rules>`_. 

Otherwise, use the :kbd:`←` and :kbd:`→` arrow keys to jump around the docs.

Companion paper
^^^^^^^^^^^^^^^^

If you like math better than code, then you might prefer to read `our paper <https://arxiv.org/abs/2405.14813>`_:

.. code::
    
    @article{modula,
      author  = {Tim Large and Yang Liu and Minyoung Huh and Hyojin Bahng and Phillip Isola and Jeremy Bernstein},
      title   = {Scalable Optimization in the Modular Norm},
      journal = {arXiv:2405.14813},
      year    = 2024
    }

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction:

   bad-scaling
   golden-rules
   history

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Theory of modules:

   theory/vector
   theory/module
   theory/atom/index
   theory/bond/index
   theory/compound/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Useful links:

   Modula codebase <https://github.com/jxbz/modula>
   Modula paper <https://arxiv.org/abs/2405.14813>
