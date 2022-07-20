# qujax

Represent a (parameterised) quantum circuit as a pure [JAX](https://github.com/google/jax) function that
takes as input any parameters of the circuit and outputs a _statetensor_. The statetensor encodes all $2^N$ amplitudes of the quantum state and can then be used
downstream for exact expectations, gradients or sampling.

A JAX implementation of a quantum circuit is useful for runtime speedups, automatic differentiation and support for GPUs/TPUs.

## Install
```
pip install qujax
```

## Parameterised quantum circuits with JAX
```python
from jax import numpy as jnp
import qujax

circuit_gates = ['H', 'Ry', 'CZ']
circuit_qubit_inds = [[0], [0], [0, 1]]
circuit_params_inds = [[], [0], []]

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



## qujax.tket
You can also generate the parameter to statetensor function from a [pytket](https://cqcl.github.io/tket/pytket/api/) circuit
```python
import pytket
import qujax

circuit = pytket.Circuit(2)
circuit.H(0)
circuit.Rz(1.3, 0)
circuit.CX(0, 1)

params_to_st = qujax.tket.tk_to_qujax(circuit)
```

