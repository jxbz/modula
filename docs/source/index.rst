Welcome to the Modula docs!
============================

Modula is a deep learning framework designed for graceful scaling. Neural networks written in Modula automatically transfer learning rate across scale. Modula can be installed by running: 

.. code-block:: console

   $ pip install modula

Purpose of the docs
^^^^^^^^^^^^^^^^^^^^

We wrote these docs with the intention of explaining both scaling theory and the design of Modula in clear and simple terms. We hope that this will help speed up deep learning optimization research.

If something is unclear, first check `the FAQ <faq>`_, but then consider starting a `GitHub issue <https://github.com/jxbz/modula/issues>`_, making a `pull request <https://github.com/jxbz/modula/pulls>`_ or reaching out to us by email. Then we can improve the docs for everyone.

Navigating the docs
^^^^^^^^^^^^^^^^^^^^

You can use the :kbd:`←` and :kbd:`→` arrow keys to jump around the docs. You can also use the side panel.

Companion paper
^^^^^^^^^^^^^^^^

If you prefer to read a more academic-style paper, then you can check out `our arXiv paper <https://arxiv.org/abs/2405.14813>`_:

.. code::
    
    @article{modula,
      author  = {Tim Large and Yang Liu and Minyoung Huh and Hyojin Bahng and Phillip Isola and Jeremy Bernstein},
      title   = {Scalable Optimization in the Modular Norm},
      journal = {arXiv:2405.14813},
      year    = 2024
    }

Acknowledgements
^^^^^^^^^^^^^^^^^

Thanks to Gavia Gray, Uzay Girit and Jyo Pari for helpful feedback.

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
   :caption: Theory of Modules:

   theory/vector
   theory/module
   theory/atom/index
   theory/bond/index
   theory/compound/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: More on Modula:

   Modula FAQ <faq>
   Modula codebase <https://github.com/jxbz/modula>
   Modula paper <https://arxiv.org/abs/2405.14813>
