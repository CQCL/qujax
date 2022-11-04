from __future__ import annotations
from typing import Sequence, Callable, Union
from jax import numpy as jnp, random
from jax.lax import fori_loop

from qujax.statetensor import apply_gate
from qujax.utils import check_hermitian, sample_integers, paulis


def statetensor_to_single_expectation(statetensor: jnp.ndarray,
                                      hermitian: jnp.ndarray,
                                      qubit_inds: Sequence[int]) -> float:
    """
    Evaluates expectation value of an observable represented by a Hermitian matrix (in tensor form).

    Args:
        statetensor: Input statetensor.
        hermitian: Hermitian array
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of qubit indices for Hermitian matrix to be applied to.
            Must have 2 * len(qubit_inds) == hermitian.ndim

    Returns:
        Expected value (float).
    """
    statetensor_new = apply_gate(statetensor, hermitian, qubit_inds)
    axes = tuple(range(statetensor.ndim))
    return jnp.tensordot(statetensor.conjugate(), statetensor_new, axes=(axes, axes)).real


def get_hermitian_tensor(hermitian_seq: Sequence[Union[str, jnp.ndarray]]) -> jnp.ndarray:
    """
    Convert a sequence of observables represented by Pauli strings or Hermitian matrices in tensor form
    into single array (in tensor form).

    Args:
        hermitian_seq: Sequence of Hermitian strings or arrays.

    Returns:
        Hermitian matrix in tensor form (array).
    """
    for h in hermitian_seq:
        check_hermitian(h)

    single_arrs = [paulis[h] if isinstance(h, str) else h for h in hermitian_seq]
    single_arrs = [h_arr.reshape((2,) * int(jnp.log2(h_arr.size))) for h_arr in single_arrs]

    full_mat = single_arrs[0]
    for single_matrix in single_arrs[1:]:
        full_mat = jnp.kron(full_mat, single_matrix)
    full_mat = full_mat.reshape((2,) * int(jnp.log2(full_mat.size)))
    return full_mat


def _get_tensor_to_expectation_func(hermitian_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                    qubits_seq_seq: Sequence[Sequence[int]],
                                    coefficients: Union[Sequence[float], jnp.ndarray],
                                    contraction_function: Callable) \
        -> Callable[[jnp.ndarray], float]:
    """
    Takes strings (or arrays) representing Hermitian matrices, along with qubit indices and
    a list of coefficients and returns a function that converts a tensor into an expected value.
    The contraction function performs the tensor contraction according to the type of tensor provided
    (i.e. whether it is a statetensor or a densitytensor).

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian matrix is either represented by a tensor (jnp.ndarray) or by a list of 'X', 'Y' or 'Z' characters corresponding to the standard Pauli matrices.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.
        contraction_function: Function that performs the tensor contraction.

    Returns:
        Function that takes tensor and returns expected value (float).
    """

    hermitian_tensors = [get_hermitian_tensor(h_seq) for h_seq in hermitian_seq_seq]

    def statetensor_to_expectation_func(statetensor: jnp.ndarray) -> float:
        """
        Maps statetensor to expected value.

        Args:
            statetensor: Input statetensor.

        Returns:
            Expected value (float).
        """
        out = 0
        for hermitian, qubit_inds, coeff in zip(hermitian_tensors, qubits_seq_seq, coefficients):
            out += coeff * contraction_function(statetensor, hermitian, qubit_inds)
        return out

    return statetensor_to_expectation_func


def get_statetensor_to_expectation_func(hermitian_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                        qubits_seq_seq: Sequence[Sequence[int]],
                                        coefficients: Union[Sequence[float], jnp.ndarray]) \
        -> Callable[[jnp.ndarray], float]:
    """
    Takes strings (or arrays) representing Hermitian matrices, along with qubit indices and
    a list of coefficients and returns a function that converts a statetensor into an expected value.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian matrix is either represented by a tensor (jnp.ndarray)
            or by a list of 'X', 'Y' or 'Z' characters corresponding to the standard Pauli matrices.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """

    return _get_tensor_to_expectation_func(hermitian_seq_seq, qubits_seq_seq, coefficients,
                                           statetensor_to_single_expectation)


def get_statetensor_to_sampled_expectation_func(hermitian_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                                qubits_seq_seq: Sequence[Sequence[int]],
                                                coefficients: Union[Sequence[float], jnp.ndarray]) \
        -> Callable[[jnp.ndarray, random.PRNGKeyArray, int], float]:
    """
    Converts strings (or arrays) representing Hermitian matrices, qubit indices and
    coefficients into a function that converts a statetensor into a sampled expected value.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian is either a tensor (jnp.ndarray) or a string in ('X', 'Y', 'Z').
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor, random key and integer number of shots
        and returns sampled expected value (float).
    """
    statetensor_to_expectation_func = get_statetensor_to_expectation_func(hermitian_seq_seq,
                                                                          qubits_seq_seq,
                                                                          coefficients)

    def statetensor_to_sampled_expectation_func(statetensor: jnp.ndarray,
                                                random_key: random.PRNGKeyArray,
                                                n_samps: int) -> float:
        """
        Maps statetensor to sampled expected value.

        Args:
            statetensor: Input statetensor.
            random_key: JAX random key
            n_samps: Number of samples contributing to sampled expectation.

        Returns:
            Sampled expected value (float).
        """
        sampled_integers = sample_integers(random_key, statetensor, n_samps)
        sampled_probs = fori_loop(0, n_samps,
                                  lambda i, sv: sv.at[sampled_integers[i]].add(1),
                                  jnp.zeros(statetensor.size))

        sampled_probs /= n_samps
        sampled_st = jnp.sqrt(sampled_probs).reshape(statetensor.shape)
        return statetensor_to_expectation_func(sampled_st)

    return statetensor_to_sampled_expectation_func
