# Generalizing Across Domains via Cross-Gradient Training

Tensorflow implementation of the paper [**Generalizing Across Domains via Cross-Gradient Training**](https://arxiv.org/abs/1804.10745).

## Usage

The file *crossgrad.py* contains the base tensorflow implementation with no dependencies (except for tensorflow itself and the *util.py* supplied in this same repository).

An example of usage can be found in the notebook *CrossGrad.ipynb*, which requires to run:

* SenseTheFlow branch v3.0 (https://github.com/gpascualg/SenseTheFlow/tree/v3.0)
* MNIST python wrappers (https://github.com/datapythonista/mnist)

## Understanding crossgrad function

In *crossgrad.py* you can find the base function for the model. Its parameters map to the paper ones as (see Figure 2, Algorithm 1):

* *labels_fn*: $C_{\theta_l}$
* *latent_fn*: $G_{\theta^1_d}$ (defaults to a simple CNN + dense network with final dimension controlled by *latent_space_dimensions*)
* *domain_fn*: $S_{\theta^2_d}$ (defaults to a simple 2-dense network)
* *$x$*: As the paper, features
* *domain*: $d$
* *labels*: $y$
* *epsilon_d*: $\epsilon_d$
* *epsilon_l*: $\epsilon_l$
* *alpha_d*: $\alpha_d$
* *alpha_l*: $\alpha_l$
* *params*: Not in the paper, some configuration (such as 'summaries' and 'data_format')
