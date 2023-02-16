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

JAX is emerging as a state-of-the-art library for high-perfomance scientific computation in Python 
due to is composability, automatic differentiation and support for GPUs/TPUs, as well as adopting 
the NumPy [@numpy] API resulting in a low barrier to entry.

`qujax` is a lightweight, purely functional library written entirely in JAX, 
composing seamlessly with the ever-expanding JAX ecosystem (e.g. @deepmindjax, @blackjax, @mocat). 
It emphasises clarity and readability, making it easy to debug, reducing the barrier to entry,
 and decreasing the overhead when integrating with existing code or extending it to meet specific
  research needs.

These characteristics contrast with the already existing array of excellent quantum computation resources in Python
supporting JAX as a backend, such as cirq [@cirq], pytket [@pytket], qiskit [@jax2018github], Qulacs [@qulacs],
TensorFlow Quantum [@tensorflowquantum], Pennylane [@pennylane] or quimb [@quimb]. 
These represent complex full-fledged frameworks which supply their own abstractions, being either wider 
in scope or specializing in specific use-cases.

There is an active area of research investigating tensor networks as a tool for classical 
simulation of quantum circuits with software including DisCoPy [@discopy], quimb [@quimb] and 
TensorCircuit [@tensorcircuit]. While tensor networks represent a very promising field of research, 
their implementation entails a more sophisticated API (in tensor networks, the representation
of a quantum state can be considerably more elaborate), greatly increasing the complexity of the
package. Thus, tensor network computation is currently seen as being beyond the scope of `qujax`.


# `pytket-qujax`

TODO: briefly explain function of pytket-qujax (convert between circuit representations,
    can run on actual device, bridges pytket and JAX ecosystems).


# Acknowledgements

We acknowledge notable support from Kirill Plekhanov as well as Gabriel Marin and Enrico Rinaldi.


# References



