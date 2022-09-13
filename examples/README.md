# qujax examples

In this directory, you can find a selection of notebooks demonstrating some simple use cases of `qujax`

- [`generative_modelling.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/generative_modelling.ipynb) - uses a parameterised quantum circuit as a generative model for a real life dataset. Trains via stochastic gradient Langevin dynamics on the maximum mean discrepancy between statetensor and dataset.
- [`heisenberg_vqe.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/heisenberg_vqe.ipynb) - an implementation of the variational quantum eigensolver to find the ground state of a quantum Hamiltonian.
- [`maxcut_vqe.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/maxcut_vqe.ipynb) - an implementation of the variational quantum eigensolver to solve a maxcut problem. Trains with Adam via [`optax`](https://github.com/deepmind/optax) and uses more realistic stochastic parameter shift gradients.
- [`qaoa.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/qaoa.ipynb) - uses a problem inspired QAOA ansatz to find the ground state of a quantum Hamiltonian. Demonstrates how to encode more sophisticated parameters that control multiple gates.
- [`variational_inference.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/variational_inference.ipynb) - uses a parameterised quantum circuit as a variational distribution to fit to a target probability mass function. Uses Adam via [`optax`](https://github.com/deepmind/optax) to minimise the KL divergence between circuit and target distributions.

The Heisenberg notebook with a `tk_to_qujax` implementation can be found in the [`pytket`](https://github.com/CQCL/pytket) repository at [`pytket-qujax_heisenberg_vqe.ipynb`](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb).
