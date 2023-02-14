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


TODO: Different simulators (statetensor, densitytensor, unitarytensor)

TODO: Observables


# Statement of need

There already exists an array of quantum computation resources in Python
including cirq [@cirq], pytket [@pytket], qiskit [@jax2018github], Qulacs [@qulacs],
TensorFlow Quantum [@tensorflowquantum] as well as 
Pennylane [@pennylane] and Quimb [@quimb] which support JAX as a backend.

However, unlike all of the aforementioned, `qujax` is written entirely in JAX and is purely functional.
JAX is emerging as the state-of-the-art library for high-perfomance scientific computation in Python 
due to is composability, automatic differentiation and support for GPUs/TPUs as well as adopting 
a NumPy [@numpy] API resulting in a low barrier to entry. Being a pure JAX library makes 
`qujax` very lightweight and therefore easy to pick up and contribute to, it also 
composes seamlessly with the ever expanding JAX ecosystem (e.g. @deepmindjax, @blackjax, @mocat). 
Additionally, the purely functional syntax of `qujax` makes it more readable as well as being 
easier to debug and compose with existing code.

There is also an active area of research investigating tensor networks as a tool for classical 
simulation of quantum circuits with software including DisCoPy [@discopy], quimb [@quimb] and 
TensorCircuit [@tensorcircuit]. Tensor networks represent a very promising field of research 
however come with a significantly more sophisticed API (since with tensor networks one is 
typically interested in a single quantity such as an amplitude or expectation value rather 
than a complete characterisation of the state as provided by `qujax`) - thus tensor network 
computation is currently seen as beyond the scope of `qujax`.


# `pytket-qujax`

TODO: briefly explain function of pytket-qujax (convert between circuit representations,
    can run on actual device, bridges pytket and JAX ecosystems).


# Acknowledgements

We acknowledge notable support from Kirill Plekhanov as well as Gabriel Marin and Enrico Rinaldi.


# References



