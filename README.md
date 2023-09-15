# qujax

[![PyPI - Version](https://img.shields.io/pypi/v/qujax)](https://pypi.org/project/qujax/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05504/status.svg)](https://doi.org/10.21105/joss.05504)

* [Installation](#installation)
* [Quick start](#quick-start)
  + [Pure state simulation](#pure-state-simulations)
  + [Mixed state simulation](#mixed-state-simulations)
* [Converting from TKET](#converting-from-tket)
* [Examples](#examples)
* [Contributing](#contributing)
* [Citing qujax](#citing-qujax)
* [API Reference](https://cqcl.github.io/qujax/api/)

qujax is a [JAX](https://github.com/google/jax)-based Python library for the classical simulation of quantum circuits. It is designed to be *simple*, *fast* and *flexible*.

It follows a functional programming design by translating circuits into pure functions. This allows qujax to [seamlessly interface with JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions), enabling direct access to its powerful automatic differentiation tools, just-in-time compiler, vectorization capabilities, GPU/TPU integration and growing ecosystem of packages.

qujax can be used both for pure and for mixed quantum state simulation. It not only supports the standard gate set, but also allows user-defined custom operations, including general quantum channels, enabling the user to e.g. model device noise and errors. 

An overview of the core functionalities of qujax can be found in the [Quick start](#quick-start) section. More advanced use-cases, including the training of parameterised quantum circuits, are listed in [Examples](#examples).


## Installation

qujax is [hosted on PyPI](https://pypi.org/project/qujax/) and can be installed via the pip package manager
```
pip install qujax
```

## Quick start

**Important note: qujax circuit parameters are expressed in units of $\pi$ (e.g. in the range $[0,2]$ as opposed to $[0, 2\pi]$)**.

### Pure state simulation

We start by defining the quantum gates making up the circuit, along with the qubits that they act on and the indices of the parameters for each gate. 

A list of all gates can be found [here](https://github.com/CQCL/qujax/blob/main/qujax/gates.py) (custom operations can be included by [passing an array or function](https://cqcl.github.io/qujax/api/get_params_to_statetensor_func.html) instead of a string).

```python
from jax import numpy as jnp
import qujax

# List of quantum gates
circuit_gates = ['H', 'Ry', 'CZ']
# Indices of qubits the gates will be applied to
circuit_qubit_inds = [[0], [0], [0, 1]]
# Indices of parameters each parameterised gate will use
circuit_params_inds = [[], [0], []]

qujax.print_circuit(circuit_gates, circuit_qubit_inds, circuit_params_inds);
# q0: -----H-----Ry[0]-----◯---
#                          |   
# q1: ---------------------CZ--
```

We then translate the circuit to a pure function `param_to_st` that takes a set of parameters and an (optional) initial quantum state as its input.

```python
param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                   circuit_qubit_inds,
                                                   circuit_params_inds)

param_to_st(jnp.array([0.1]))
# Array([[0.58778524+0.j, 0.        +0.j],
#        [0.80901706+0.j, 0.        +0.j]], dtype=complex64)
```

The optional initial state can be passed to `param_to_st` using the `statetensor_in` argument. When it is not provided, the initial state defaults to $\ket{0...0}$.

Note that qujax represents quantum states as _statetensors_. For example, for $N=4$ qubits, the corresponding vector space has $2^4$ dimensions, and a quantum state in this space is represented by an array with shape `(2,2,2,2)`. The usual statevector representation with shape `(16,)` can be obtained by calling `.flatten()` or `.reshape(-1)` or `.reshape(2**N)` on this array. 

In the statetensor representation, the coefficient associated with e.g. basis state $\ket{0101}$ is given by `arr[0,1,0,1]`; each axis corresponds to one qubit.

```python
param_to_st(jnp.array([0.1])).flatten()
# Array([0.58778524+0.j, 0.+0.j, 0.80901706+0.j, 0.+0.j], dtype=complex64)
```

Finally, by defining an observable, we can map the statetensor to an expectation value. A general observable is specified using lists of Pauli matrices, the qubits they act on, and the associated coefficients. 

For example, $Z_1Z_2Z_3Z_4 - 2 X_3$ would be written as `[['Z','Z','Z','Z'], ['X']], [[1,2,3,4], [3]], [1., -2.]`.

```python
st_to_expectation = qujax.get_statetensor_to_expectation_func([['Z']], [[0]], [1.])
```

Combining `param_to_st` and `st_to_expectation` gives us a parameter to expectation function that can be automatically differentiated using JAX.

```python
from jax import value_and_grad

param_to_expectation = lambda param: st_to_expectation(param_to_st(param))
expectation_and_grad = value_and_grad(param_to_expectation)
expectation_and_grad(jnp.array([0.1]))
# (Array(-0.3090171, dtype=float32),
#  Array([-2.987832], dtype=float32))
```

### Mixed state simulation
Mixed state simulations are analogous to the above, but with calls to `get_params_to_densitytensor_func` and `get_densitytensor_to_expectation_func` instead.

```python
param_to_dt = qujax.get_params_to_densitytensor_func(circuit_gates,
                                                     circuit_qubit_inds,
                                                     circuit_params_inds)
dt = param_to_dt(jnp.array([0.1]))
dt.shape
# (2, 2, 2, 2)

dt_to_expectation = qujax.get_densitytensor_to_expectation_func([['Z']], [[0]], [1.])
dt_to_expectation(dt)
# Array(-0.3090171, dtype=float32)
```

Similarly to a statetensor, which represents the reshaped $2^N$-dimensional statevector of a pure quantum state, a _densitytensor_ represents the reshaped $2^N \times 2^N$ density matrix of a mixed quantum state. This densitytensor has shape `(2,) * 2 * N`.

For example, for $N=2$, and a mixed state $\frac{1}{2} (\ket{00}\bra{11} + \ket{11}\bra{00} + \ket{11}\bra{11} + \ket{00}\bra{00})$, the corresponding densitytensor `dt` is such that `dt[0,0,1,1] = dt[1,1,0,0] = dt[1,1,1,1] = dt[0,0,0,0] = 1/2`, and all other entries are zero.

The equivalent density matrix can be obtained by calling `.reshape(2 ** N, 2 ** N)`.

## Converting from TKET

One can directly convert a [`pytket`](https://cqcl.github.io/tket/pytket/api/) circuit using the [`tk_to_qujax`](https://cqcl.github.io/pytket-qujax/api/api.html#pytket.extensions.qujax.qujax_convert.tk_to_qujax) and [`tk_to_qujax_symbolic`](https://cqcl.github.io/pytket-qujax/api/api.html#pytket.extensions.qujax.qujax_convert.tk_to_qujax_symbolic) functions in the [**`pytket-qujax`**](https://github.com/CQCL/pytket-qujax) extension.

An example of this can be found in the [`pytket-qujax_heisenberg_vqe.ipynb`](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb) notebook.

## Examples

Below are some use-case notebooks. These both illustrate the flexibility of qujax and the power of directly interfacing with JAX and its package ecosystem.

- [`heisenberg_vqe.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/heisenberg_vqe.ipynb) - an implementation of the variational quantum eigensolver to find the ground state of a quantum Hamiltonian.
- [`maxcut_vqe.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/maxcut_vqe.ipynb) - an implementation of the variational quantum eigensolver to solve a MaxCut problem. Trains with Adam via [`optax`](https://github.com/deepmind/optax) and uses more realistic stochastic parameter shift gradients.
- [`noise_channel.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/noise_channel.ipynb) - uses the densitytensor simulator to fit the parameters of a depolarising noise channel.
- [`qaoa.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/qaoa.ipynb) - uses a problem-inspired QAOA ansatz to find the ground state of a quantum Hamiltonian. Demonstrates how to encode more sophisticated parameters that control multiple gates.
- [`variational_inference.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/variational_inference.ipynb) - uses a parameterised quantum circuit as a variational distribution to fit to a target probability mass function. Uses Adam via [`optax`](https://github.com/deepmind/optax) to minimise the KL divergence between circuit and target distributions.
- [`classification.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/classification.ipynb) - train a quantum circuit for binary classification using data re-uploading.
- [`generative_modelling.ipynb`](https://github.com/CQCL/qujax/blob/main/examples/generative_modelling.ipynb) - uses a parameterised quantum circuit as a generative model for a real life dataset. Trains via stochastic gradient Langevin dynamics on the maximum mean discrepancy between statetensor and dataset.

The [`pytket`](https://github.com/CQCL/pytket) repository also contains `tk_to_qujax` implementations for some of the above at [`pytket-qujax_classification.ipynb`](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax-classification.ipynb), 
[`pytket-qujax_heisenberg_vqe.ipynb`](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb) 
and [`pytket-qujax_qaoa.ipynb`](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_qaoa.ipynb).


## Contributing

You can open a bug report or a feature request by creating a new [issue on GitHub](https://github.com/CQCL/qujax/issues).

Pull requests are welcome! To open a new one, please go through the following steps:

1. First fork the repo and create your branch from [`develop`](https://github.com/CQCL/qujax/tree/develop).
2. Commit your code and tests.
4. Update the documentation, if required.
5. Check the code lints (run `black . --check` and `pylint */`).
6. Issue a pull request into the [`develop`](https://github.com/CQCL/qujax/tree/develop) branch.

New commits on [`develop`](https://github.com/CQCL/qujax/tree/develop) will be merged into
[`main`](https://github.com/CQCL/qujax/tree/main) in the next release.


## Citing qujax

If you have used qujax in your code or research, we kindly ask that you cite it. You can use the following BibTeX entry for this:

```bibtex
@article{qujax2023,
  author = {Duffield, Samuel and Matos, Gabriel and Johannsen, Melf},
  doi = {10.21105/joss.05504},
  journal = {Journal of Open Source Software},
  month = sep,
  number = {89},
  pages = {5504},
  title = {{qujax: Simulating quantum circuits with JAX}},
  url = {https://joss.theoj.org/papers/10.21105/joss.05504},
  volume = {8},
  year = {2023}
}
```