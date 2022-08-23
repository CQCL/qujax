# qujax

Represent a (parameterised) quantum circuit as a pure [JAX](https://github.com/google/jax) function that
takes as input any parameters of the circuit and outputs a _statetensor_. The statetensor encodes all $2^N$ amplitudes of the quantum state and can then be used
downstream for exact expectations, gradients or sampling.

A JAX implementation of a quantum circuit is useful for runtime speedups, automatic differentiation and support for GPUs/TPUs.

Some useful links:
- [Documentation](https://cqcl.github.io/qujax/api/)
- [PyPI](https://pypi.org/project/qujax/)
- [Example notebooks](https://github.com/CQCL/qujax/tree/main/examples)
- [pytket-qujax](https://github.com/CQCL/pytket-qujax)


## Install
```
pip install qujax
```

## Parameterised quantum circuits with qujax
```python
from jax import numpy as jnp
import qujax

circuit_gates = ['H', 'Ry', 'CZ']
circuit_qubit_inds = [[0], [0], [0, 1]]
circuit_params_inds = [[], [0], []]

qujax.print_circuit(circuit_gates, circuit_qubit_inds, circuit_params_inds)
# q0: -----H-----Ry[0]-----◯---
#                          |   
# q1: ---------------------CZ--
```

```python
param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                   circuit_qubit_inds,
                                                   circuit_params_inds)
```

We now have a pure JAX function that generates the statetensor for given parameters
```python
param_to_st(jnp.array([0.1]))
# DeviceArray([[0.58778524+0.j, 0.        +0.j],
#              [0.80901706+0.j, 0.        +0.j]], dtype=complex64)
```

The statevector can be obtained from the statetensor via ```.flatten()```.
```python
param_to_st(jnp.array([0.1])).flatten()
# DeviceArray([0.58778524+0.j, 0.+0.j, 0.80901706+0.j, 0.+0.j], dtype=complex64)
```

We can also use qujax to map the statetensor to an expected value
```python
st_to_expectation = qujax.get_statetensor_to_expectation_func([['Z']], [[0]], [1.])
```

Combining the two gives us a parameter to expectation function that can be differentiated seamlessly and exactly with JAX
```python
from jax import value_and_grad

param_to_expectation = lambda param: st_to_expectation(param_to_st(param))
expectation_and_grad = value_and_grad(param_to_expectation)
expectation_and_grad(jnp.array([0.1]))
# (DeviceArray(-0.3090171, dtype=float32),
#    DeviceArray([-2.987832], dtype=float32))
```


## Notes
+ We use the convention where parameters are given in units of π (i.e. in [0,2] rather than [0, 2π]).
+ By default the parameter to statetensor function initiates in the all 0 state, however there is an optional ```statetensor_in``` argument to initiate in an arbitrary state.


## pytket-qujax
You can also generate the parameter to statetensor function from a [`pytket`](https://cqcl.github.io/tket/pytket/api/)
circuit using the [`pytket-qujax`](https://github.com/CQCL/pytket-qujax) extension.
In particular, the
[`tk_to_qujax`](https://cqcl.github.io/pytket-qujax/api/api.html#pytket.extensions.qujax.qujax_convert.tk_to_qujax) and
[`tk_to_qujax_symbolic`](https://cqcl.github.io/pytket-qujax/api/api.html#pytket.extensions.qujax.qujax_convert.tk_to_qujax_symbolic)
functions.
An example notebook can be found at [`pytket-qujax_heisenberg_vqe.ipynb`](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb).


## Contributing
Bugs and feature requests are managed using [GitHub issues](https://github.com/CQCL/qujax/issues).

Pull requests are welcomed!
1. First fork the repo and create your branch from [`develop`](https://github.com/CQCL/qujax/tree/develop).
2. Add your code.
3. Add your tests.
4. Update the documentation if required.
5. Issue a pull request into [`develop`](https://github.com/CQCL/qujax/tree/develop).

New commits on [`develop`](https://github.com/CQCL/qujax/tree/develop) will then be merged into
[`main`](https://github.com/CQCL/qujax/tree/main) on the next release.
