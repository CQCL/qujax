from __future__ import annotations
from typing import Sequence, Union, Callable, Protocol

from jax import numpy as jnp

from qujax import gates
from qujax.circuit_tools import check_circuit


class CallableArrayAndOptionalArray(Protocol):
    def __call__(self, params: jnp.ndarray, statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        ...


class CallableOptionalArray(Protocol):
    def __call__(self, statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        ...


UnionCallableOptionalArray = Union[CallableArrayAndOptionalArray, CallableOptionalArray]


def _get_apply_gate(gate_func: Callable,
                    qubit_inds: Sequence[int]) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Creates a function that applies a given gate_func to given qubit_inds of a statetensor.

    Args:
        gate_func: Function that takes any gate parameters and returns the gate unitary (in tensor form).
        qubit_inds: Sequence of indices for gate to be applied to.
            len(qubit_inds) is equal to the dimension of the gate unitary tensor.

    Returns:
        Function that takes statetensor and gate parameters, returns updated statetensor.

    """

    def apply_gate(statetensor: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """
        Applies {} gate to statetensor.

        Args:
            statetensor: Input statetensor.
            params: Gate parameters if any.

        Returns:
            Updated statetensor.
        """
        gate_unitary = gate_func(*params)
        statetensor = jnp.tensordot(gate_unitary, statetensor,
                                    axes=(list(range(-len(qubit_inds), 0)), qubit_inds))
        statetensor = jnp.moveaxis(statetensor, list(range(len(qubit_inds))), qubit_inds)
        return statetensor

    apply_gate.__doc__ = apply_gate.__doc__.format(gate_func.__name__)

    return apply_gate


def get_params_to_statetensor_func(gate_seq: Sequence[Union[str,
                                                            jnp.ndarray,
                                                            Callable[[jnp.ndarray], jnp.ndarray],
                                                            Callable[[], jnp.ndarray]]],
                                   qubit_inds_seq: Sequence[Sequence[int]],
                                   param_inds_seq: Sequence[Sequence[int]],
                                   n_qubits: int = None) -> UnionCallableOptionalArray:
    """
    Creates a function that maps circuit parameters to a statetensor.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching an array or function in qujax.gates,
            a unitary array (which will be reshaped into a tensor of shape (2,2,2,...) )
            or a function taking parameters and returning gate unitary in tensor form.
        qubit_inds_seq: Sequences of qubits (ints) that gates are acting on.
        param_inds_seq: Sequence of parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the first parameter,
            the second gate is not parameterised and the third gates used the fifth and second parameters.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps parameters (and optional statetensor_in) to a statetensor.
        If no parameters are found then the function only takes optional statetensor_in.

    """

    check_circuit(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    def _array_to_callable(arr: jnp.ndarray) -> Callable[[], jnp.ndarray]:
        return lambda: arr

    gate_seq_callable = []
    for gate in gate_seq:
        if isinstance(gate, str):
            gate = gates.__dict__[gate]

        if callable(gate):
            gate_func = gate
        elif hasattr(gate, '__array__'):
            gate_arr = jnp.array(gate)
            gate_size = gate_arr.size
            gate = gate_arr.reshape((2,) * int(jnp.log2(gate_size)))
            gate_func = _array_to_callable(gate)
        else:
            raise TypeError(f'Unsupported gate type - gate must be either a string in qujax.gates, an array or '
                            f'callable: {gate}')
        gate_seq_callable.append(gate_func)

    apply_gate_seq = [_get_apply_gate(g, q) for g, q in zip(gate_seq_callable, qubit_inds_seq)]
    param_inds_seq = [jnp.array(p) for p in param_inds_seq]
    param_inds_seq = [jnp.array([]) if jnp.any(jnp.isnan(p)) else p.astype(int) for p in param_inds_seq]

    def params_to_statetensor_func(params: jnp.ndarray,
                                   statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        """
        Applies parameterised circuit (series of gates) to a statetensor_in (default is |0>^N).

        Args:
            params: Parameters of the circuit.
            statetensor_in: Optional. Input statetensor.
                Defaults to |0>^N (tensor of size 2^n with all zeroes except one in [0]*N index).

        Returns:
            Updated statetensor.

        """
        if statetensor_in is None:
            statetensor = jnp.zeros((2,) * n_qubits)
            statetensor = statetensor.at[(0,) * n_qubits].set(1.)
        else:
            statetensor = statetensor_in
        params = jnp.atleast_1d(params)
        for gate_ind, apply_gate in enumerate(apply_gate_seq):
            gate_params = jnp.take(params, param_inds_seq[gate_ind])
            statetensor = apply_gate(statetensor, gate_params)
        return statetensor

    if all([pi.size == 0 for pi in param_inds_seq]):
        def no_params_to_statetensor_func(statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
            """
            Applies circuit (series of gates with no parameters) to a statetensor_in (default is |0>^N).

            Args:
                statetensor_in: Optional. Input statetensor.
                    Defaults to |0>^N (tensor of size 2^n with all zeroes except one in [0]*N index).

            Returns:
                Updated statetensor.

            """
            return params_to_statetensor_func(jnp.array([]), statetensor_in)

        return no_params_to_statetensor_func

    return params_to_statetensor_func

