# Examples

Below are some use-case notebooks. These both illustrate the flexibility of qujax and the power of directly interfacing with JAX and its package ecosystem.

- [heisenberg_vqe.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/heisenberg_vqe.ipynb) - an implementation of the variational quantum eigensolver to find the ground state of a quantum Hamiltonian.
- [maxcut_vqe.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/maxcut_vqe.ipynb) - an implementation of the variational quantum eigensolver to solve a MaxCut problem. Trains with Adam via [`optax`](https://github.com/deepmind/optax) and uses more realistic stochastic parameter shift gradients.
- [noise_channel.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/noise_channel.ipynb) - uses the densitytensor simulator to fit the parameters of a depolarising noise channel.
- [qaoa.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/qaoa.ipynb) - uses a problem-inspired QAOA ansatz to find the ground state of a quantum Hamiltonian. Demonstrates how to encode more sophisticated parameters that control multiple gates.
- [barren_plateaus.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/barren_plateaus.ipynb) - illustrates how to sample gradients of a cost function to identify the presence of barren plateaus. Uses batched/vectorized evaluation to speed up computation.
- [reducing_jit_compilation_time.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/reducing_jit_compilation_time.ipynb) - explains how JAX compilation works and how that can lead to excessive compilation times when executing quantum circuits. Presents a solution for the case of circuits with a repeating structure.
- [variational_inference.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/variational_inference.ipynb) - uses a parameterised quantum circuit as a variational distribution to fit to a target probability mass function. Uses Adam via [`optax`](https://github.com/deepmind/optax) to minimise the KL divergence between circuit and target distributions.
- [classification.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/classification.ipynb) - train a quantum circuit for binary classification using data re-uploading.
- [generative_modelling.ipynb](https://github.com/CQCL/qujax/blob/develop/examples/generative_modelling.ipynb) - uses a parameterised quantum circuit as a generative model for a real life dataset. Trains via stochastic gradient Langevin dynamics on the maximum mean discrepancy between statetensor and dataset.

The [pytket](https://github.com/CQCL/pytket) repository also contains `tk_to_qujax` implementations for some of the above at [pytket-qujax_classification.ipynb](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax-classification.ipynb), 
[pytket-qujax_heisenberg_vqe.ipynb](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb) 
and [pytket-qujax_qaoa.ipynb](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_qaoa.ipynb).