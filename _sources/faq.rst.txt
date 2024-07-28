Frequently asked questions
===========================

Feel free to reach out or start a `GitHub issue <https://github.com/jxbz/modula/issues>`_ if you have any questions about Modula. We'll post answers to any useful or common questions on this page.

.. dropdown:: The gradient is a vector: how can a vector have a spectral norm?
	:icon: question

	An important mental jump in Modula is to think of the weights of our neural network as a list of tensors :math:`(\mathbf{W}_1, \dots \mathbf{W}_L)` where :math:`\mathbf{W}_k` is the weight tensor of layer :math:`k`. It then makes sense to think of the gradient of the loss :math:`\mathcal{L}` with respect to the :math:`k\text{th}` weight tensor :math:`\mathbf{W}_k` as itself being a tensor :math:`\nabla_{\mathbf{W_k}}\mathcal{L}` with the same shape as :math:`\mathbf{W}_k`. We can then meaningfully ask what is the operator norm of this gradient tensor.

	This contrasts with a common approach to optimization theory where the whole weight space is "flattened" into one big weight vector :math:`\mathbf{w}` with a corresponding gradient vector :math:`\nabla_\mathbf{w} \mathcal{L}`, thus "losing" the operator structure.

.. dropdown:: Why does Adam beat SGD on transformer, and how does normalization fix SGD?
	:icon: question

	While some researchers `have challenged <https://arxiv.org/abs/1705.08292>`_ the use of Adam in deep learning, Adam is certainly the optimizer of choice for training large language models, `performing much better <https://arxiv.org/abs/2407.07972>`_ than SGD in practice. Still, it is not widely known *why* Adam is better than SGD. Here we aim to provide a mechanistic explanation of one of the main reasons. The basic idea is that there is no reason the raw gradients should have good relative sizes across layers. And a major thing that Adam does is to "rebalance" the update sizes across layers.

	Let's give a concrete example to see what we mean. Consider a machine learning model with a list of weight tensors :math:`\mathbf{w} = (\mathbf{W}_1, \dots \mathbf{W}_L)` and a loss function :math:`\mathcal{L}`. Then a vanilla gradient update is given by :math:`(\mathbf{W}_1, \dots \mathbf{W}_L) - \eta \times (\nabla_{\mathbf{W}_1}\mathcal{L}, \dots \nabla_{\mathbf{W}_L}\mathcal{L})` where :math:`\eta` is the global learning rate. Now, suppose that our neural network is a toy residual network with :math:`L` layers:

	.. math::
		f(\mathbf{w} ;\mathbf{x}) := \mathbf{W}_L \left(1 + \frac{1}{L} \mathbf{W_{L-1}}\right) \dots \left(1 + \frac{1}{L} \mathbf{W_{2}}\right) \mathbf{W_1} \mathbf{x}.

	This toy network consists of "read-in" and "read-out" matrices :math:`\mathbf{W}_0` and :math:`\mathbf{W}_L` along with :math:`L-2` "residual" matrices each depressed by a factor of :math:`1/L`. These depression factors are included to give the model a better large depth limit---in Modula we advocate for :math:`1/L` depression factors, while the the inclusion of :math:`1/\sqrt{L}` depression factors is `standard in large language models <https://proceedings.mlr.press/v119/huang20f.html>`_. We do not include a nonlinearity in this toy model for simplicity.

	The point is that the depression factors---be they :math:`1/L` or :math:`1/\sqrt{L}`---also depress the gradients to the residual blocks by the same factor. So if one takes the depth :math:`L` large and uses vanilla gradient descent or SGD to train a transformer, one is essentially applying the update:

	.. math::

		(\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3, \dots \mathbf{W}_{L-2}, \mathbf{W}_{L-1}, \mathbf{W}_L) - \eta \times (\nabla_{\mathbf{W}_1}\mathcal{L}, 0, 0, \dots, 0, 0,  \nabla_{\mathbf{W}_L}\mathcal{L}).

	In words: the inclusion of the depression factors kills the size of the updates to the residual blocks in comparison to the read-in and read-out layers in deep networks. If you use SGD to train such a model, depending on how you set the learning rate :math:`\eta`, you are stuck between severely under-training the middle layers or severely over-training the input and output layers. Adam largely fixes this issue by normalizing each update tensor individually and thus removing the effect of the depression factors. So, Adam is a form of gradient normalization! Modular normalization also automatically fixes this issue by rebalancing the size of the updates for any base optimizer.

.. dropdown:: Why does modular normalization lead to learning rate transfer across scale?
	:icon: question

	By the definition of a "well-normed module" :math:`\mathsf{M}`, when weight updates :math:`\Delta \mathbf{w}` are normalized in the modular norm :math:`\|\cdot\|_\mathsf{M}` then updates :math:`\Delta \mathbf{y}` to the module output are well-behaved in the output norm :math:`\|\cdot\|_\mathcal{Y}`. We set up our actual architectures, including complicated models like GPT, to actually be well-normed independent of the scale of the architecture. A little bit more formally:

	1. well-normed modules are one-Lipschitz in the modular norm, meaning :math:`\|\Delta \mathbf{y}\|_\mathcal{Y} \leq \|\Delta \mathbf{w}\|_\mathsf{M}`;
	2. this inequality holds tightly when tensors in the network "align" during training, meaning that we may approximate :math:`\|\Delta \mathbf{y}\|_\mathcal{y} \approx \|\Delta \mathbf{w}\|_\mathsf{M}` in a fully aligned network;
	3. therefore normalizing updates in the modular norm provides control on the change in outputs; 
	4. these statements are all independent of the size of the architecture.

	Since modular normalization works by recursively normalizing the weight updates to each submodule, these desirable properties extend to all submodules as well as the overall compound.

.. dropdown:: What do we mean by "tensor alignment" in Modula?
	:icon: question

	In the guts of a neural network there can be found lots and lots of tensors. And sometimes these tensors like to multiply each other. For example, there are:

	- vector-vector products :math:`\mathbf{u}^\top\mathbf{v}`
	- matrix-vector products :math:`\mathbf{A}\mathbf{v}`
	- matrix-matrix products :math:`\mathbf{A}\mathbf{B}`
	- and so on...
	
	An important question is "how big are such tensor products inside a neural network?" In other words, if we know the size of the inputs to the product, can we predict the size of the product itself?

	Let's start with the simplest example of the vector-vector product, otherwise known as a friendly "dot product". Suppose we have two :math:`n` dimensional vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` of known sizes :math:`\|\mathbf{u}\|_2` and :math:`\|\mathbf{v}\|_2`. Here the symbol :math:`\|\mathbf{\cdot}\|_2` denotes the "Euclidean length" or ":math:`\ell_2` norm" of the vectors. How large can the dot product be?
	Well, by the Cauchy-Schwarz inequality, we have that:

	.. math::

		|\mathbf{u}^\top \mathbf{v}| \leq \|\mathbf{u}\|_2 \times \|\mathbf{v}\|_2.

	In words: the size of the dot product is limited by the size of its two inputs. What's more the Cauchy-Schwarz inequality is "tight", meaning that :math:`|\mathbf{u}^\top \mathbf{v}| = \|\mathbf{u}\|_2 \times \|\mathbf{v}\|_2`, when the two vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` point in the same (or opposite) directions---when the two vectors "align".

	This idea of having an inequality that limits the size of a tensor product, which is tight under certain configurations of the input tensors, generalizes to higer-order forms of tensor product. For example, for the matrix-vector product :math:`\mathbf{A}\mathbf{v}` the relevant inequality is given by:

	.. math::

		\|\mathbf{A} \mathbf{v}\|_2 \leq \|\mathbf{A}\|_* \times \|\mathbf{v}\|_2,

	where :math:`\|\cdot\|_*` is the matrix spectral norm. This inequality is tight when the vector :math:`\mathbf{v}` lies in the top singular subspace of the matrix :math:`\mathbf{A}`---when the matrix and vector "align". 

	And for matrix-matrix products, we have the "sub-multiplicativity of the spectral norm":

	.. math::

		\|\mathbf{A} \mathbf{B}\|_* \leq \|\mathbf{A}\|_* \times \|\mathbf{B}\|_*.

	We will say that this inequality is tight when the two matrices "align"---you get the idea!

	Why does any of this matter? Well for a neural network at initialization, some of these inequalities may be quite slack because tensors in the network are randomly oriented with respect to each other. But it is a central tenet of the Modula framework that after training has sufficiently "warmed up", the network will fall into a fully aligned state where all inequalities of the type mentioned in the section hold reasonably tightly, and may therefore be used to predict the size and scaling of various quantities in the network.

	.. admonition:: Other notions of alignment
	   :class: seealso

	   We have outlined a notion of alignment which captures whether or not a certain inequality governing a tensor product is tight. This is different to the notion of alignment measured in `Scaling Exponents Across Parameterizations and Optimizers <https://arxiv.org/abs/2407.05872>`_ which `turns out to be coupled to the matrix stable rank <https://x.com/jxbz/status/1814289986885140614>`_. Essentially, the findings on alignment in that paper don't have an obvious bearing on the notion of alignment used in Modula. Large-scale empirical tests of alignment as we have described it are certainly a valuable direction for future work.

.. dropdown:: The modular norm involves a max---why do I not see any maxes in the package?
	:icon: question

	Coming soon.

.. dropdown:: Is there a unique optimal way to parameterize an architecture?
	:icon: question

	Coming soon.

.. dropdown:: What is the difference between Modula and μP?
	:icon: question

	Coming soon.

.. dropdown:: Is it necessary to use orthogonal intialization in Modula?
	:icon: question

	No. You could re-write the atomic modules to use Gaussian initialization if you wanted. The reason we choose to use orthogonal initialization is that it makes it much easier to get scaling right. This is because the spectral norm of any :math:`m \times n` random orthogonal matrix is always one. In contrast, the spectral norm of an :math:`m \times n` random Gaussian matrix depends on the dimensions :math:`m` and :math:`n` and also the entry-wise variance :math:`\sigma^2`, making it more difficult to properly set the initialization scale. In addition, orthogonal matrices have the benign property that all singular values are one. In Gaussian matrices, on the other hand, the average singular value and the max singular value are different, meaning that Gaussian matrices have more subtle numerical properties.

.. dropdown:: Does Modula support weight sharing?
	:icon: question

	Not yet, although we plan to implement this and provide some examples.