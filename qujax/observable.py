from typing import Sequence, Callable, Union

import jax.numpy as jnp

from qujax import gates


def _statetensor_to_single_expectation_func(gate_tensor: jnp.ndarray,
                                            qubit_inds: Sequence[int]) -> Callable[[jnp.ndarray], float]:
    """
    Creates a function that maps statetensor to its expected value under the given gate unitary and qubit indices.

    Args:
        gate_tensor: Gate unitary in tensor form.
        qubit_inds: Sequence of integer qubit indices to apply gate to.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """
    def statetensor_to_single_expectation(statetensor: jnp.ndarray) -> float:
        """
        Evaluates expected value of statetensor through gate.

        Args:
            statetensor: Input statetensor.

        Returns:
            Expected value (float).
        """
        statetensor_new = jnp.tensordot(gate_tensor, statetensor,
                                        axes=(list(range(-len(qubit_inds), 0)), qubit_inds))
        statetensor_new = jnp.moveaxis(statetensor_new, list(range(len(qubit_inds))), qubit_inds)
        axes = tuple(range(statetensor.ndim))
        return jnp.tensordot(statetensor.conjugate(), statetensor_new, axes=(axes, axes)).real

    return statetensor_to_single_expectation


def get_statetensor_to_expectation_func(gate_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                        qubits_seq_seq: Sequence[Sequence[int]],
                                        coefficients: Union[Sequence[float], jnp.ndarray])\
        -> Callable[[jnp.ndarray], float]:
    """
    Converts gate strings, qubit indices and coefficients into a function that converts statetensor into expected value.

    Args:
        gate_seq_seq: Sequence of sequences of gates.
            Each gate is either or tensor (jnp.ndarray) or a string corresponding to gates in qujax.gates.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """

    def get_gate_tensor(gate_seq: Sequence[str]) -> jnp.ndarray:
        """
        Convert sequence of gate strings into single gate unitary (in tensor form).

        Args:
            gate_str_seq: Sequence of gate strings.

        Returns:
            Single gate unitary in tensor form (array).

        """
        single_gate_matrices = [gates.__dict__[gate]() if isinstance(gate, str) else gate for gate in gate_seq]
        full_gate_mat = single_gate_matrices[0]
        for single_gate_matrix in single_gate_matrices[1:]:
            full_gate_mat = jnp.kron(full_gate_mat, single_gate_matrix)
        full_gate_mat = full_gate_mat.reshape((2,) * (jnp.log2(full_gate_mat.size)).astype(int))
        return full_gate_mat

    apply_gate_funcs = [_statetensor_to_single_expectation_func(get_gate_tensor(gns), qi)
                        for gns, qi in zip(gate_seq_seq, qubits_seq_seq)]

    def statetensor_to_expectation_func(statetensor: jnp.ndarray) -> float:
        """
        Maps statetensor to expected value.

        Args:
            statetensor: Input statetensor.

        Returns:
            Expected value (float).

        """
        out = 0
        for coeff, f in zip(coefficients, apply_gate_funcs):
            out += coeff * f(statetensor)
        return out

    return statetensor_to_expectation_func
