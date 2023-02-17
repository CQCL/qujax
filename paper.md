---
title: 'qujax: Simulating quantum circuits with JAX'
tags:
  - Python
  - JAX
  - quantum computation
authors:
  - name: Samuel Duffield
    orcid: 0000-0002-8656-8734
    corresponding: true
    affiliation: 1
  - name: Gabriel Matos
    affiliation: "1, 2"
    orcid: 0000-0002-3373-0128
  - name: Melf Johannsen
    affiliation: 1
affiliations:
  - name: Quantinuum
    index: 1
  - name: University of Leeds
    index: 2

date: 9 February 2023
bibliography: paper.bib
---

# Summary
`qujax` is a pure JAX [@jax2018github], purely functional Python package for the classical
simulation of quantum circuits. A JAX implementation of quantum circuits inherits benefits
such as seamless automatic differentiation, support for GPUs/TPUs as well as integration with
a host of other tools within the JAX ecosystem.

`qujax` is hosted on [PyPI](https://pypi.org/project/qujax/) for easy installation and comes with detailed
[documentation](https://cqcl.github.io/qujax/api/) and a suite of
[example notebooks](https://github.com/CQCL/qujax/tree/main/examples).


`qujax` represents a quantum circuit by a collection of three native Python iterables: a series of gate
identifiers (either a string pointing to an object in `qujax.gates`, a unitary array or a function that
outputs a unitary array), a series indicating which qubits to apply the gates to and finally a series
of parameter indices indicating where, if any, to extract parameters for said gate from a single
parameter vector.

In quantum mechanics, a *pure state* is fully specified by a statevector
$|\psi\rangle = \sum_{i=1}^{2^N} \alpha_i |i\rangle \in \mathbb{C}^{2^N}$ where $n$ is the number 
of qubits and $\alpha_i$ is the $i$th *amplitude*. We assume the computational basis where 
$|i\rangle$ is a vector of zeros and a one in the $i$th position. In `qujax`, we adopt the 
*statetensor* notation where a pure state is encoded in a tensor of complex numbers with 
shape `(2,) * N`. The statetensor representation is convenient for quantum arithmetic (such as 
applying gates and sampling bitstrings) and the statevector can always be obtained by 
calling `statevector = statetensor.flatten()`. One can then use 
`qujax.get_params_to_statetensor_func` to generate a pure JAX function encoding a parameterised 
quantum state $|\psi_\theta \rangle = U_\theta |\phi\rangle$ where $\theta$ is a parameter 
vector and $|\phi\rangle$ is an optional `statetensor_in` (that defaults to $|0\rangle$). 
Alternatively, one can call `qujax.get_params_to_unitarytensor_func` to get a tensor version of 
the unitary $U_\theta$ with shape `(2,) * 2 * N`.

A second representation of quantum states is that of *density matrices*. A density matrix 
can store a quantum state via the outer product 
$\rho = |\psi \rangle \langle \psi| \in \mathbb{C}^{2^N \times 2^N}$, but more 
generally a density matrix encodes a *mixed state* 
$\rho = \sum_{k} p_k|\psi_k \rangle \langle \psi_k|$ (a classical mixture of pure states with 
$p_k \in [0,1]$ and $\sum_{k} p_k =1$). Density matrices are also supported in `qujax` in the form 
of a *densitytensors* - complex tensors of shape `(2,) * 2 * N`. Similar to the statetensor 
simulator, parameterised evolution of a densitytensor can be implemented via very general Kraus 
operations with `qujax.get_params_to_densitytensor_func`.

Expectation values can also be calculated conveniently with `qujax`. In simple cases, such as 
combinatorial optimisation like maxcut, this can be done by extracting measurement probabilities 
from the statetensor or densitytensor and calculating the expected value of a cost function. For 
more sophisticated bases, `qujax.get_statetensor_to_expectation_func` and 
`qujax.get_densitytensor_to_expectation_func` generate functions that map to the expected value 
of a given series of Hermitian tensors.


# Statement of need

JAX is emerging as a state-of-the-art library for high-perfomance scientific computation in Python 
due to is composability, automatic differentiation and support for GPUs/TPUs, as well as adopting 
the NumPy [@numpy] API resulting in a low barrier to entry.

`qujax` is a lightweight, purely functional library written entirely in JAX, 
composing seamlessly with the ever-expanding JAX ecosystem (e.g. @deepmindjax, @blackjax, @mocat). 
It emphasises clarity and readability, making it easy to debug, reducing the barrier to entry,
 and decreasing the overhead when integrating with existing code or extending it to meet specific
  research needs.

These characteristics contrast with the already existing array of excellent quantum computation resources in Python, such as cirq [@cirq], pytket [@pytket], qiskit [@jax2018github], Qulacs [@qulacs],
TensorFlow Quantum [@tensorflowquantum], Pennylane [@pennylane] or quimb [@quimb], 
the latter two supporting JAX as a backend. 
These represent complex full-fledged frameworks which supply their own abstractions, being either wider 
in scope or specializing in specific use-cases.

There is an active area of research investigating tensor networks as a tool for classical 
simulation of quantum circuits with software including DisCoPy [@discopy], quimb [@quimb] and 
TensorCircuit [@tensorcircuit]. While tensor networks represent a very promising field of research, 
their implementation entails a more sophisticated API (in tensor networks, the representation
of a quantum state can be considerably more elaborate), greatly increasing the complexity of the
package. Thus, tensor network computation is currently seen as being beyond the scope of `qujax`.


# `pytket-qujax`

`qujax` is accompanied by an extension package `pytket-qujax` supporting easy conversion to and 
from `pytket.Circuit` objects, thus providing a convenient bridge between pytket and JAX ecosystems.


# Acknowledgements

We acknowledge notable support from Kirill Plekhanov as well as Gabriel Marin and Enrico Rinaldi.


# References



