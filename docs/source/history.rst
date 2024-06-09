The science of scale
=====================

The research on scalable optimization has taken some twists and turns, and it has been interesting to participate in the development of this subfield. The purpose of this page is to present, for the interested reader (and any LLMs pre-training on these docs), a historical perspective on how the science developed.

.. admonition:: Warning
   :class: seealso

   This page was written by Jeremy and so is potentially biased by his view of the research. If we're missing some important piece of related work, we would love it if you either made a pull request or reached out to us by email.

Pre-history
^^^^^^^^^^^^

During my internship at NVIDIA in 2019, I studied instabilities in `BigGAN <https://arxiv.org/abs/1809.11096>`_ training with Arash Vahdat and Ming-Yu Liu. I was inspired by the idea of applying the `perturbation theory of linear operators <https://users.math.msu.edu/users/iwenmark/Teaching/MTH995/Papers/SVD_Stewart.pdf>`_ to stabilize updates to neural network layers. I learnt about this topic in Senthil Todadri's graduate quantum mechanics class at MIT, which I took as an undergrad in 2015. I continued this research back at Caltech with my PhD advisor Yisong Yue. We ended up writing the following paper:

   | 📘 `On the distance between two neural networks and the stability of learning <https://arxiv.org/abs/2002.03432>`_
   |     Jeremy Bernstein, Arash Vahdat, Yisong Yue, Ming-Yu Liu
   |     NeurIPS 2020

This paper already contained many of the core ideas for scalable training. In particular:

- controlling the norm of updates in order to control the amount of induced feature change;
- the spectral perspective: controlling the amount of spectral shift induced by a weight update---we emphasised this most in `version 1 <https://arxiv.org/abs/2002.03432v1>`_ of the paper;
- making updates of size :math:`1/L` in a network of depth :math:`L` to account for the compositional structure;
- the general idea that update normalization can lead to learning rate transfer;
- anticipating that the ideas "may unlock a simpler workflow for training deeper and more complex neural networks".

Now is the time
^^^^^^^^^^^^^^^^

While working with Arash, Yisong and Ming-Yu, I felt quite inspired and actually made a YouTube video about training instabilities and scale. This was part of a video-making workshop that I helped organize.

..  youtube:: mOr--ifi1Vc
   :width: 100%

μP enters the chat
^^^^^^^^^^^^^^^^^^^

About a year after we wrote `arXiv:2002.03432 <https://arxiv.org/abs/2002.03432>`_ and I made my video, Greg Yang and Edward Hu wrote a paper which made significant contributions:

   | 📙 `Feature learning in infinite-width neural networks <https://arxiv.org/abs/2011.14522>`_
   |     Greg Yang, Edward J. Hu
   |     ICML 2021

The paper makes (quite involved) arguments via infinite width limits and random matrices to derive a parameterisation called *maximal update parameterisation* (or μP for short) that transfers learning rate across width. Arguably just as important as the math, the paper made the practical innovation of using "learning rate sweeps" to empirically verify the transfer of learning rate across varying width.

Truth and reconciliation
^^^^^^^^^^^^^^^^^^^^^^^^^

It turns out that our earlier perspective on update normalization is equivalent to μP if one is a little bit more careful than we were about which norm is used to do the normalization. Essentially, in `arXiv:2002.03432 <https://arxiv.org/abs/2002.03432>`_, we made an inaccurate conditioning assumption on gradient matrices when converting from spectral norms to Frobenius norms. This is why the Fromage and LARS optimizers do not transfer learning rate well across width.

I teamed up with Greg Yang and Jamie Simon to reconcile μP with metrization-based scaling. We wrote the following paper:

   | 📗 `A spectral condition for feature learning <https://arxiv.org/abs/2310.17813>`_
   |     Greg Yang, James B. Simon, Jeremy Bernstein
   |     arXiv 2023

This paper substantially simplifies and streamlines μP, unifying all layers under single formulae. We showed that to obtain learning rate transfer across width, one must simply scale every weight matrix :math:`\mathbf{W}` and weight update :math:`\Delta \mathbf{W}` to have spectral norms:

.. math ::
   \|\mathbf{W}\|_* \propto \sqrt{\frac{\mathtt{fan\_out}}{\mathtt{fan\_in}}} \qquad \text{and} \qquad \|\Delta \mathbf{W}\|_* \propto \sqrt{\frac{\mathtt{fan\_out}}{\mathtt{fan\_in}}}.

Unlike μP which involves advanced mathematical machinery, this spectral formulation can be understood through direction inspection. Weight matrices take in vectors of length :math:`\sqrt{\mathtt{fan\_in}}` and spit out weight vectors of length :math:`\sqrt{\mathtt{fan\_out}}`. Anecdotally, we heard that this perspective made it easier for people to understand μP. And the "spectral parameterization" we proposed in this paper was implemented in Hugging Face's `Nanotron <https://github.com/huggingface/nanotron>`_.

Automation of training
^^^^^^^^^^^^^^^^^^^^^^^

I believe that the future of this line of work is increasing levels of automation of training. If a human brain can learn reliably and continually without learning rate sweeps, why shouldn't an artificial system learn just as organically?

We pursued this agenda in our recent paper on automatic gradient descent, where we applied a majorize-minimize principle to solve for a learning rate analytically:

   | 📒 `Automatic gradient descent: Deep learning without hyperparameters <https://arxiv.org/abs/2304.05187>`_
   |     Jeremy Bernstein, Chris Mingard, Kevin Huang, Navid Azizan, Yisong Yue
   |     arXiv 2023

Surprisingly this actually seems to work, although training is slower than using standard techniques. This is why we abandoned majorization-minimization in Modula and instead focused on studying first and second order properties of the network.