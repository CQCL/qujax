from __future__ import annotations
from typing import Sequence, Union, Callable, Iterable, Tuple
from jax import numpy as jnp
from jax.lax import scan

from qujax.circuit import apply_gate, UnionCallableOptionalArray, gate_type
from qujax.circuit import _to_gate_func, _arrayify_inds, _gate_func_to_unitary
from qujax.circuit_tools import check_circuit

kraus_op_type = Union[gate_type, Iterable[gate_type]]

def statetensor_to_densitytensor(statetensor: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a densitytensor representation of a pure quantum state
    from its statetensor representaton

    Args:
        statetensor: Input statetensor.

    Returns:
        A densitytensor representing the quantum state.
    """
    n_qubits = statetensor.ndim
    st = statetensor
    dt = (st.reshape(-1, 1) @ st.reshape(1, -1).conj()).reshape(2 for _ in range(2*n_qubits))
    return dt


def _kraus_single(densitytensor: jnp.ndarray,
                  array: jnp.ndarray,
                  qubit_inds: Sequence[int]) -> jnp.ndarray:
    """
    Performs single Kraus operation

    .. math::
        \rho_\text{out} = B \rho_\text{in} B^{\dagger}

    Args:
        densitytensor: Input density matrix of shape=(2, 2, ...) and ndim=2*n_qubits
        array: Array containing the Kraus operator (in tensor form).
        qubit_inds: Sequence of qubit indices on which to apply the Kraus operation.

    Returns:
        Updated density matrix.
    """
    n_qubits = densitytensor.ndim // 2
    densitytensor = apply_gate(densitytensor, array, qubit_inds)
    densitytensor = apply_gate(densitytensor, array.conj(), [n_qubits + i for i in qubit_inds])
    return densitytensor


def kraus(densitytensor: jnp.ndarray,
          arrays: Iterable[jnp.ndarray],
          qubit_inds: Sequence[int]) -> jnp.ndarray:
    """
    Performs Kraus operation.

    .. math::
        \rho_\text{out} = \sum_i B_i \rho_\text{in} B_i^{\dagger}

    Args:
        densitytensor: Input density matrix of shape=(2, 2, ...) and ndim=2*n_qubits
        arrays: Sequence of arrays containing the Kraus operators (in tensor form).
        qubit_inds: Sequence of qubit indices on which to apply the Kraus operation.

    Returns:
        Updated density matrix.
    """
    arrays = jnp.array(arrays)
    if arrays.ndim % 2 == 0:
        arrays = arrays[jnp.newaxis]
        # ensure first dimensions indexes different kraus operators
    arrays = arrays.reshape((arrays.shape[0],) + (2,) * 2 * len(qubit_inds))

    new_densitytensor, _ = scan(lambda dt, arr: (dt + _kraus_single(densitytensor, arr, qubit_inds), None),
                                init=jnp.zeros_like(densitytensor) * 0.j, xs=arrays)
    # i.e. new_densitytensor = vmap(_kraus_single, in_axes=(None, 0, None))(densitytensor, arrays, qubit_inds).sum(0)
    return new_densitytensor


def _to_kraus_operator_seq_funcs(kraus_op: kraus_op_type,
                                 param_inds: Union[None, Sequence[int], Sequence[Sequence[int]]]) \
        -> Tuple[Sequence[Callable[[jnp.ndarray], jnp.ndarray]],
                 Sequence[jnp.ndarray]]:
    """
    Ensures Kraus operators are a sequence of functions that map (possibly empty) parameters to tensors
    and that each element of param_inds_seq is a sequence of arrays that correspond to the parameter indices
    of each Kraus operator.

    Args:
        kraus_op: Either a normal gate_type or a sequence of gate_types representing Kraus operators.
        param_inds: If kraus_op is a normal gate_type then a sequence of parameter indices,
            if kraus_op is a sequence of Kraus operators then a sequence of sequences of parameter indices

    Returns:
        Tuple containing sequence of functions mapping to Kraus operators
        and sequence of arrays with parameter indices

    """
    if param_inds is None:
        param_inds = [None for _ in kraus_op]

    if isinstance(kraus_op, (list, tuple)):
        kraus_op_funcs = [_to_gate_func(ko) for ko in kraus_op]
    else:
        kraus_op_funcs = [_to_gate_func(kraus_op)]
        param_inds = [param_inds]
    return kraus_op_funcs, _arrayify_inds(param_inds)


def partial_trace(densitytensor: jnp.ndarray,
                  indices_to_trace: Iterable[int]) -> jnp.ndarray:
    """
    Traces out (discards) specified qubits, resulting in a densitytensor
    representing the mixed quantum state on the remaining qubits.

    Args:
        densitytensor: Input densitytensor.
        indices_to_trace: Indices of qubits to trace out/discard.

    Returns:
        Resulting densitytensor on remaining qubits.

    """
    n_qubits = densitytensor.ndim // 2
    einsum_indices = list(range(densitytensor.ndim))
    for i in indices_to_trace:
        einsum_indices[i + n_qubits] = einsum_indices[i]
    densitytensor = jnp.einsum(densitytensor, einsum_indices)
    return densitytensor


def get_params_to_densitytensor_func(kraus_ops_seq: Sequence[kraus_op_type],
                                     qubit_inds_seq: Sequence[Sequence[int]],
                                     param_inds_seq: Sequence[Union[None, Sequence[int], Sequence[Sequence[int]]]],
                                     n_qubits: int = None) -> UnionCallableOptionalArray:
    """
    Creates a function that maps circuit parameters to a density tensor.
    densitytensor = densitymatrix.reshape((2,) * 2 * n_qubits)
    densitymatrix = densitytensor.reshape(2 ** n_qubits, 2 ** n_qubits)

    Args:
        kraus_ops_seq: Sequence of gates.
            Each element is either a string matching a unitary array or function in qujax.gates,
            a custom unitary array or a custom function taking parameters and returning a unitary array.
            Unitary arrays will be reshaped into tensor form (2, 2,...)
        qubit_inds_seq: Sequences of sequences representing qubit indices (ints) that gates are acting on.
            i.e. [[0], [0,1], [1]] tells qujax the first gate is a single qubit gate acting on the zeroth qubit,
            the second gate is a two qubit gate acting on the zeroth and first qubit etc.
        param_inds_seq: Sequence of sequences representing parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the zeroth parameter
            (the float at position zero in the parameter vector/array), the second gate is not parameterised
            and the third gates used the parameters at position five and two.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps parameters (and optional densitytensor_in) to a densitytensor.
        If no parameters are found then the function only takes optional densitytensor_in.

    """

    check_circuit(kraus_ops_seq, qubit_inds_seq, param_inds_seq, n_qubits, False)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    kraus_ops_seq_callable_and_param_inds = [_to_kraus_operator_seq_funcs(ko, param_inds)
                                             for ko, param_inds in zip(kraus_ops_seq, param_inds_seq)]
    kraus_ops_seq_callable = [ko_pi[0] for ko_pi in kraus_ops_seq_callable_and_param_inds]
    param_inds_array_seq = [ko_pi[1] for ko_pi in kraus_ops_seq_callable_and_param_inds]

    def params_to_densitytensor_func(params: jnp.ndarray,
                                     densitytensor_in: jnp.ndarray = None) -> jnp.ndarray:
        """
        Applies parameterised circuit (series of gates) to a densitytensor_in (default is |0>^N <0|^N).

        Args:
            params: Parameters of the circuit.
            densitytensor_in: Optional. Input densitytensor.
                Defaults to |0>^N <0|^N (tensor of size 2^(2*N) with all zeroes except one in [0]*(2*N) index).

        Returns:
            Updated densitytensor.

        """
        if densitytensor_in is None:
            densitytensor = jnp.zeros((2,) * 2 * n_qubits)
            densitytensor = densitytensor.at[(0,) * 2 * n_qubits].set(1.)
        else:
            densitytensor = densitytensor_in
        params = jnp.atleast_1d(params)
        for gate_func_single_seq, qubit_inds, param_inds_single_seq in zip(kraus_ops_seq_callable, qubit_inds_seq,
                                                                           param_inds_array_seq):
            kraus_operators = [_gate_func_to_unitary(gf, qubit_inds, pi, params)
                               for gf, pi in zip(gate_func_single_seq, param_inds_single_seq)]
            densitytensor = kraus(densitytensor, kraus_operators, qubit_inds)
        return densitytensor

    non_parameterised = all([all([pi.size == 0 for pi in pi_seq]) for pi_seq in param_inds_array_seq])
    if non_parameterised:
        def no_params_to_densitytensor_func(densitytensor_in: jnp.ndarray = None) -> jnp.ndarray:
            """
            Applies circuit (series of gates with no parameters) to a densitytensor_in (default is |0>^N <0|^N).

            Args:
                densitytensor_in: Optional. Input densitytensor.
                    Defaults to |0>^N <0|^N (tensor of size 2^(2*N) with all zeroes except one in [0]*(2*N) index).

            Returns:
                Updated densitytensor.

            """
            return params_to_densitytensor_func(jnp.array([]), densitytensor_in)

        return no_params_to_densitytensor_func

    return params_to_densitytensor_func


def densitytensor_to_measurement_probabilities(densitytensor: jnp.ndarray,
                                               qubit_inds: Sequence[int]) -> jnp.ndarray:
    """
    Extract array of measurement probabilities given a densitytensor and some qubit indices to measure
    (in the computational basis).
    I.e. the ith element of the array corresponds to the probability of observing the bitstring
    represented by the integer i on the measured qubits.

    Args:
        densitytensor: Input densitytensor.
        qubit_inds: Sequence of qubit indices to measure.

    Returns:
        Normalised array of measurement probabilities.
    """
    n_qubits = densitytensor.ndim // 2
    n_qubits_measured = len(qubit_inds)
    qubit_inds_trace_out = [i for i in range(n_qubits) if i not in qubit_inds]
    return jnp.diag(partial_trace(densitytensor, qubit_inds_trace_out).reshape(2 * n_qubits_measured,
                                                                               2 * n_qubits_measured)).real


def densitytensor_to_measured_densitytensor(densitytensor: jnp.ndarray,
                                            qubit_inds: Sequence[int],
                                            measured_int: int) -> jnp.ndarray:
    """
    Returns the post-measurement densitytensor assuming that qubit_inds are measured
    (in the computational basis) and the bitstring corresponding to integer
    measured_int is observed.

    Args:
        densitytensor: Input densitytensor.
        qubit_inds: Sequence of qubit indices to measure.
        measured_int: Observed integer.

    Returns:
        Post-measurement densitytensor (same shape as input densitytensor).
    """
    n_qubits = densitytensor.ndim // 2
    n_qubits_measured = len(qubit_inds)
    qubit_inds_projector = jnp.diag(jnp.zeros(2 ** n_qubits_measured).at[measured_int].set(1)) \
        .reshape((2,) * 2 * n_qubits_measured)
    unnorm_densitytensor = _kraus_single(densitytensor, qubit_inds_projector, qubit_inds)
    norm_const = jnp.trace(unnorm_densitytensor.reshape(2**n_qubits,  2**n_qubits)).real
    return unnorm_densitytensor / norm_const
