Frequently asked questions
===========================

Feel free to reach out or start a `GitHub issue <https://github.com/jxbz/modula/issues>`_ if you have any questions about Modula. We'll post answers to any useful or common questions on this page.

.. dropdown:: Why does Adam beat SGD on transformer, and why does normalization fix SGD?
	:icon: question

	Coming soon.

.. dropdown:: Why does modular normalization lead to learning rate transfer across scale?
	:icon: question

	In simple terms, when weight updates :math:`\Delta \mathbf{w}` are normalized in the modular norm :math:`\|\cdot\|_\mathsf{M}` of the module :math:`\mathsf{M}` then updates :math:`\Delta \mathbf{y}` to the module output are well-behaved in the output norm :math:`\|\cdot\|_\mathcal{Y}`, independent of the scale of the architecture. A little bit more formally:

	1. modules are one-Lipschitz in the modular norm, meaning that :math:`\|\Delta \mathbf{y}\|_\mathcal{Y} \leq \|\Delta \mathbf{w}\|_\mathsf{M}`;
	2. this inequality holds tightly when tensors in the network align during training, meaning that :math:`\|\Delta \mathbf{y}\|_\mathcal{y} \approx \|\Delta \mathbf{w}\|_\mathsf{M}` in a fully aligned network;
	3. therefore normalizing updates in the modular norm provides control on the change in outputs, independent of the size of the architecture.

	Since modular normalization works by recursively normalizing the weight updates to each submodule, these desirable properties in fact extend to all submodules as well as the overall compound.

.. dropdown:: What do we mean by "tensor alignment" in Modula?
	:icon: question

	Coming soon.

.. dropdown:: The modular norm involves a max---why do I not see any maxes in the package?
	:icon: question

	Coming soon.

.. dropdown:: Is it necessary to use orthogonal intialization in Modula?
	:icon: question

	No. You could re-write the atomic modules to use Gaussian initialization if you wanted. The reason we choose to use orthogonal initialization is that it makes it much easier to get scaling right. This is because the spectral norm of any :math:`m \times n` random orthogonal matrix is always one. In contrast, the spectral norm of an :math:`m \times n` random Gaussian matrix depends on the dimensions :math:`m` and :math:`n` and also the entry-wise variance :math:`\sigma^2`, making it more difficult to properly set the initialization scale. In addition, orthogonal matrices have the benign property that all singular values are one. In Gaussian matrices, on the other hand, the average singular value and the max singular value are different, meaning that Gaussian matrices have more subtle numerical properties.

.. dropdown:: Does Modula support weight sharing?
	:icon: question

	Not yet, although we plan to implement this and provide some examples.