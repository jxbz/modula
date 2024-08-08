Frequently asked questions
===========================

Feel free to reach out or start a `GitHub issue <https://github.com/jxbz/modula/issues>`_ if you have any questions about Modula. We'll post answers to any useful or common questions on this page.

Conceptual questions
^^^^^^^^^^^^^^^^^^^^^

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

.. dropdown:: Is there a unique and optimal way to parameterize an architecture?
	:icon: question

	The short answer is no: if you're careful, there is some freedom in how you can parameterize your architecture. With that said, there are some constraints that you can't really avoid if you want things to work well. And there are some "natural choices" which I think we may as well agree on at least to ease communication between researchers.

	A `LoRA layer <https://arxiv.org/abs/2106.09685>`_ provides a really good setting to think about these points. Given a :math:`n \times r` matrix :math:`B` and an :math:`r \times n` matrix :math:`A`, a LoRA layer is just the matrix product :math:`B A`. Now if you're a `spectral-μP <https://arxiv.org/abs/2310.17813>`_ afficionado, you'd know that the "right way" to scale these matrices is so that their initialization and updates have spectral norm proportional to :math:`\sqrt{\text{fan-out/fan-in}}`. Written out in full:

	- matrix :math:`B` and update :math:`\Delta B` have spectral norms :math:`\|B\|_*` and :math:`\|\Delta B\|_* \propto \sqrt{n / r}`,
	- matrix :math:`A` and update :math:`\Delta A` have spectral norms :math:`\|A\|_*` and :math:`\|\Delta A\|_* \propto \sqrt{r / n}`.

	However, these conditions are more restrictive than necessary. Because matrices are homogeneuous linear maps, in the product :math:`BA` we are free to scale up the matrix :math:`B` by any factor so long as we divide the matrix :math:`A` by the same factor. Nothing changes if we do this. In particular, if we scale :math:`B` by factor :math:`\sqrt{r/n}` and divide :math:`A` by this same factor we obtain new conditions:

	- matrix :math:`B` and update :math:`\Delta B` have spectral norms :math:`\|B\|_*` and :math:`\|\Delta B\|_* \propto 1`,
	- matrix :math:`A` and update :math:`\Delta A` have spectral norms :math:`\|A\|_*` and :math:`\|\Delta A\|_* \propto 1`.

	Using these new spectral scaling conditions will have exactly the same training dynamics.

	.. admonition:: Matters of precision
	   :class: seealso

	   When considering representing the weight entries in floating point, a difference may emerge between these two schemes. In particular, one scheme may lead to weight entries more easily representable in a low-precision floating point number format. Charlie Blake et al. consider exploiting this type of "scale symmetry" in `u-μP: The Unit-Scaled Maximal Update Parametrization <https://arxiv.org/abs/2407.17465>`_.

	In summary, I hope that this section demonstrates that:

	1. the conditions in the spectral-μP paper provide a sensible default way of scaling matrices which should work well in generic situations;
	2. however, the conditions are not unique, and in specific cases you can modify the rules---so long as you know what you're doing;
	3. you may want to take advantage of scale symmetries if you are interested in designing low-precision training algorithms.

Related work
^^^^^^^^^^^^^

.. dropdown:: What is the relationship between Modula and spectral-μP?
	:icon: question

	In the `spectral-μP paper <https://arxiv.org/abs/2310.17813>`_, we considered the problem of equipping individual layers---such as linear and embedding layers---with their "natural norm". Normalizing updates in this "natural norm" leads to learning rate transfer across the dimensions of that layer. You can see Modula as generalizing this approach to arbitrary compositions and concatenations of individual layers---i.e. neural nets.

.. dropdown:: What is the relationship between Modula and Tensor Programs?
	:icon: question

	We pointed out in the section on `the science of scale <../history>`_ that Modula builds on an approach to learning rate transfer `that we first released <https://arxiv.org/abs/2002.03432>`_ almost a year before `the first incarnation of μP <https://arxiv.org/abs/2011.14522>`_. So I want to focus here on explaining the technical differences between Modula and `Tensor Programs <https://thegregyang.com>`_.

	The main advantages of Modula over Tensor Programs are that:

	1. **Modula is grounded in elementary math.** We show that learning rate transfer is essentially just the question of how to build neural nets with tight and non-dimensional Lipschitz estimates. The main ingredient is just bounding derivatives and tracking how derivative bounds behave under composition and concatenation. We do not employ limiting or probabilistic analyses.
	2. **Modula theory is non-asymptotic.** The unifying thread through the Tensor Programs series of works is the study of neural network computation in limiting cases: infinite width, infinite depth, and so on. This means that the theory is encumbered by significant mathematical overhead, and one is often confronted with thorny technical questions---for example: `do width and depth limits commute? <https://arxiv.org/abs/2302.00453>`_ In contrast, Modula is based on a completely non-asymptotic theory. It deals directly with the finite-sized neural networks that we actually use in practice, so you don't have to worry that certain technical details may be "lost in the limit". To show that this is not just talk, in our paper we `built a theory of an actual working transformer <https://arxiv.org/abs/2405.14813>`_.
	3. **Modula is more automatic.** In Modula, we automatically build a norm during construction of the computation graph that can be used to explicitly normalize weight updates taken from any base optimizer. The Tensor Programs approach essentially amounts to manually deriving a priori estimates on the size of this norm, and using these estimates to modify the SGD learning rate per layer. However, working out these prior estimates is quite a hairy procedure which seemingly does not always work, hence why later Tensor Programs papers `shift to modifying Adam updates <https://arxiv.org/abs/2308.01814>`_. Adam updates are easier to deal with since they already impose a form of normalization on the gradients. Furthermore, the Tensor Programs calculations must be done by hand. The result is large tables of scaling rules, with tables of rules for different base optimizers (Adam versus SGD) and even tables for different matrix shapes (square versus wide rectangular versus skinny rectangular).

	4. **Modula is easier to extend.** Ultimately, we hope that Modula---and more generally the idea of *metrized deep learning*---will inspire followup work on clean, simple and technically sound approaches to algorithm design in deep learning. We give some directions for future work towards the end of `our paper <https://arxiv.org/abs/2405.14813>`_, and we believe it should be relatively easy to extend our approach to handle new modules types and new norms. To give an example, there is a natural extension of the linear module that equips the input and output spaces with the :math:`\ell_\infty` norm instead of the RMS norm, thereby inducing the :math:`\ell_\infty`--:math:`\ell_\infty` operator norm on the matrix space. While from the point of view of infinite width limits, it may not matter whether you are stabilizing infinity norms or RMS norms, we believe this sort of consideration might be quite interesting in practice.

.. dropdown:: What is the relationship between Modula and AGD?
	:icon: question

	In part, Modula builds on the analysis from our previous paper on `automatic gradient descent <https://arxiv.org/abs/2304.05187>`_. The AGD paper focused on building a majorize-minimize-style analysis of deep fully-connected networks. The surprising aspect of the AGD algorithm was that it could train various deep learning problems with no learning rate, weight decay, momentum or schedule hyperparameters. However, the training was slower and sometimes not quite as good as conventional training setups.

	The Modula paper, in contrast, shows how to modularize and automate the types of technical calculations done in the AGD paper. In Modula, we conduct these calculations to first and second order, since we came to believe that a full majorization is overly pessimistic, contributing to the slower training of AGD. And ultimately in the Modula experiments, we opted to use a linear decay learning rate schedule for its simplicity and high performance, rather than various automatic learning rate schedules that could be derived from the the Modula theory.

	I (Jeremy) still think an analogue of AGD that is also fast and performant might still be possible. It might involve combining Modula with ideas from people like Konstantin Mishchenko and Aaron Defazio such as `Prodigy <https://arxiv.org/abs/2306.06101>`_ or `schedule-free optimizer <https://arxiv.org/abs/2405.15682>`_. I think this is a great direction for future work.

.. dropdown:: What is the relationship between Modula and Shampoo?
	:icon: question

	Actually no one asked this one, and I just thought about it for myself. But here goes... Consider a loss function :math:`\mathcal{L} : \mathbb{R}^{m \times n}\to\mathbb{R}`. In other words, we have a machine learning model whose weights are given by an :math:`m \times n` matrix :math:`\mathbf{W}`. Further, suppose that the loss is smooth in the sense that:

	.. math::

		\mathcal{L}(\mathbf{W} + \mathbf{\Delta W}) \leq \mathcal{L}(\mathbf{W}) +\mathrm{trace}(\mathbf{G}^\top \mathbf{\Delta W}) + \frac{1}{2} \|\mathbf{\Delta W}\|_*^2.

	In words: the loss is "smooth in the spectral norm". This toy problem is interesting for us to think about since the modular norm on linear atomic modules *is* the spectral norm. The second term on the righthand side is the `Frobenius inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`_ and :math:`\|\cdot\|_*` denotes the `spectral norm <https://mathworld.wolfram.com/SpectralNorm.html>`_. We have adopted the shorthand :math:`\mathbf{G}` for the gradient of the loss :math:`\nabla_\mathbf{W} \mathcal{L}(\mathbf{W})`, and we suppose that the gradient admits the singular value decomposition :math:`\mathbf{G} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top`.

	If we minimise the righthand side of this inequality with respect to :math:`\mathbf{\Delta W}`, we find that the optimal step direction is given by :math:`\mathbf{\Delta W} \propto - \mathbf{U}\mathbf{V}^\top`. That is, we take the negative gradient and set all of its singular values to one. This direction "squeezes the most juice" out of the gradient under a spectral norm geometry. This is a somewhat classical observation. For instance, it appears in a 2015 paper on `stochastic spectral descent <https://ieeexplore.ieee.org/document/7347351>`_. Tim independently pointed this out to me in the course of the Modula project, and so did Laker Newhouse who is a talented undergrad at MIT.

	What stopped us experimenting further with this idea is that it's not obvious how to compute :math:`\mathbf{\Delta W} \propto - \mathbf{U}\mathbf{V}^\top` without computing SVDs, and SVDs are kind of expensive in PyTorch. But a cool realization I had recently is that there is another way to compute :math:`\mathbf{U}\mathbf{V}^\top`. In fact, it holds that:

	.. math::
		\mathbf{U}\mathbf{V}^\top = (\mathbf{G}\mathbf{G}^\top)^{-\tfrac{1}{4}} \mathbf{G} (\mathbf{G}^\top \mathbf{G})^{-\tfrac{1}{4}}.

	Why is this interesting? Well, for one, that expression on the right-hand side is precisely the `Shampoo <https://arxiv.org/abs/1802.09568>`_ preconditioner with the accumulation dropped. It suggests a new perspective on Shampoo as doing "steepest descent under the spectral norm". This is a squarely "first-order" interpretation, as opposed to the predominant way people seem to think of Shampoo as an "approximate second-order method". 

	Another reason this could be interesting is that a lot of efficiencies developed for Shampoo could now be applied to `stochastic spectral descent <https://ieeexplore.ieee.org/document/7347351>`_ and in turn Modula linear modules. One of the coolest examples is something I found in `Rohan Anil's slides <https://rosanneliu.com/dlctfs/dlct_210312.pdf>`_ on Shampoo. It's the idea that you can compute expressions like :math:`(\mathbf{G}\mathbf{G}^\top)^{-1/4}` using `Newton-Raphson iterations <https://en.wikipedia.org/wiki/Newton%27s_method>`_---a very different approach to taking SVDs. A classic paper on this topic is called `On the Computation of the Matrix k-th Root <https://onlinelibrary.wiley.com/doi/10.1002/%28SICI%291521-4001%28199803%2978%3A3%3C167%3A%3AAID-ZAMM167%3E3.0.CO%3B2-R>`_ by Slobodan Lakić. `I implemented Algorithm 1 from that paper as a gist <https://gist.github.com/jxbz/fe235ee1c72b8b41ccd0d02b43378cf2>`_, finding that it often gives significant speedups over the SVD, provided one is willing to tolerate some error.

	An extremely natural way to combine this idea with Modula is to write a new "ShampooLinear" atomic module which replaces the normalize function of our Linear atom with a zeroth matrix power. Jack Gallagher has started experimenting with this idea in `Modulax <https://github.com/GallagherCommaJack/modulax/>`_.







Modula package
^^^^^^^^^^^^^^^

.. dropdown:: The modular norm involves a max---why do I not see any maxes in the package?
	:icon: question

	Computing the modular norm involves evaluating lots of expressions of the form:

	.. math::
		\| (\mathbf{w}_1, \mathbf{w}_2) \|_{\mathsf{M}} := \max ( p * \|\mathbf{w}_1\|_{\mathsf{M}_1} , q * \|\mathbf{w}_2\|_{\mathsf{M}_2}).


	So you might be surprised not to see lots of maxes in the package. This is because to normalize a vector :math:`(\mathbf{w}_1, \mathbf{w}_2)` we do not just compute :math:`(\mathbf{w}_1, \mathbf{w}_2) / \|(\mathbf{w}_1, \mathbf{w}_2)\|_\mathsf{M}`. Instead, we separately normalize both sub-vectors in order to "saturate" the max. That is, we send:

	.. math::
		(\mathbf{w}_1, \mathbf{w}_2) \mapsto \left(\frac{\mathbf{w}_1}{p * \|\mathbf{w}_1\|_{\mathsf{M}_1}}, \frac{\mathbf{w}_2}{q * \|\mathbf{w}_2\|_{\mathsf{M}_2}} \right).

	In other words, we maximize the size of each subvector under the constraint that the full vector has unit modular norm.

.. dropdown:: Is it necessary to use orthogonal intialization in Modula?
	:icon: question

	No. You could re-write the atomic modules to use Gaussian initialization if you wanted. The reason we choose to use orthogonal initialization is that it makes it much easier to get scaling right. This is because the spectral norm of any :math:`m \times n` random orthogonal matrix is always one. In contrast, the spectral norm of an :math:`m \times n` random Gaussian matrix depends on the dimensions :math:`m` and :math:`n` and also the entry-wise variance :math:`\sigma^2`, making it more difficult to properly set the initialization scale. In addition, orthogonal matrices have the benign property that all singular values are one. In Gaussian matrices, on the other hand, the average singular value and the max singular value are different, meaning that Gaussian matrices have more subtle numerical properties.

.. dropdown:: Does Modula support weight sharing?
	:icon: question

	Not at present, although it would be possible to support this.

Research philosophy
^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Do I need to be a mathematical savant to contribute to research of this kind?
	:icon: question

	I don't think so. There are a lot of very technical people working in this field bringing with them some quite advanced tools from math and theoretical physics, and this is great. But in my experience it's usually the simpler and more elementary ideas that actually work in practice. I strongly believe that deep learning theory is still at the stage of model building. And I resonate with both Rahimi and Recht's call for `simple theorems and simple experiments <https://archives.argmin.net/2017/12/11/alchemy-addendum/>`_ and George Dahl's call for `a healthy dose of skepticism <https://www.youtube.com/watch?v=huTx3rtv8q8>`_ when evaluating claims in the literature.