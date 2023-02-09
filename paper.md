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
`qujax` is pure `JAX` [@jax2018github], purely functional Python package for the classical
simulation of quantum circuits. A `JAX` implementation of quantum circuits inherits benefits
such as seamless automatic differentiation, support for GPUs/TPUs as well as integration with
a host of other tools within the `JAX` ecosystem.

`qujax` is hosted on [PyPI](https://pypi.org/project/qujax/) for easy installation and comes with detailed
[documentation](https://cqcl.github.io/qujax/api/) and a suite of
[example notebooks](https://github.com/CQCL/qujax/tree/main/examples).


`qujax` represents a quantum circuit by a collection of three Python iterables: a series of gate
identifiers (either a string pointing to an object in `qujax.gates`, a unitary array or a function that
outputs a unitary array), a series indicating which qubits to apply the gates to and finally a series
of parameter indices indicating where, if any, to extract parameters for said gate from a single
parameter vector.


TODO: Different simulators (statetensor, densitytensor, unitarytensor)

TODO: Observables

TODO: Explain why no tensor networks (they represent different paradigm, more involved API)


# Statement of need

TODO: Benefits over Qiskit, Pennylane (which also has JAX backend) - simpler, functional


# `pytket-qujax`

TODO: briefly explain function of pytket-qujax (convert between circuit representations,
    can run on actual device, bridges pytket and JAX ecosystems).


# Acknowledgements

We acknowledge notable support from Kirill Plekhanov as well as Gabriel Marin and Enrico Rinaldi.


# References





