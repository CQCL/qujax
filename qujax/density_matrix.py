
from __future__ import annotations
from string import ascii_lowercase

from typing import Sequence, Union, Callable, Iterable
import jax
from jax import numpy as jnp
from jax.lax import scan

from qujax.circuit import apply_gate, UnionCallableOptionalArray, _to_gate_funcs, _arrayify_inds
from qujax.circuit_tools import check_circuit


def _kraus_single(densitytensor: jnp.ndarray,
                  array: jnp.ndarray,
                  qubit_inds: Sequence[int]) -> jnp.ndarray:
    """
    Performs single Kraus operation

    .. math::
        \rho_\text{out} = B \rho_\text{in} B^{\dagger}

    Args:
        densitytensor: Input density matrix of shape=(2, 2, ...) and ndim=2*n_qubits
        array: Array containing the Kraus operator.
        qubit_inds: Sequence of qubit indices on which to apply the Kraus operation.

    Returns:
        Updated density matrix.
    """
    n_qubits = densitytensor.ndim // 2
    densitytensor = apply_gate(densitytensor, array, qubit_inds)
    densitytensor = apply_gate(densitytensor, array.conj(), [n_qubits + i for i in qubit_inds])
    return densitytensor


def kraus(densitytensor: jnp.ndarray,
          arrays: Union[Sequence[jnp.ndarray], jnp.ndarray],
          qubit_inds: Sequence[int]) -> jnp.ndarray:
    """
    Performs Kraus operation.

    .. math::
        \rho_\text{out} = \sum_i B_i \rho_\text{in} B_i^{\dagger}

    Args:
        densitytensor: Input density matrix of shape=(2, 2, ...) and ndim=2*n_qubits
        arrays: Sequence of arrays containing the Kraus operators.
        qubit_inds: Sequence of qubit indices on which to apply the Kraus operation.

    Returns:
        Updated density matrix.
    """
    arrays = jnp.atleast_3d(arrays)
    new_densitytensor, _ = scan(lambda dt, arr: dt + _kraus_single(densitytensor, arr, qubit_inds),
                                 init=jnp.zeros_like(densitytensor), xs=arrays)
    # i.e. new_densitytensor = vmap(_kraus_single, in_axes=(None, 0, None))(densitytensor, arrays, qubit_inds)
    return new_densitytensor

def _partial_trace(densitytensor: jnp.ndarray,
                   indices_to_contract: Iterable[int]):
    n_qubits = densitytensor.ndim // 2
    einsum_indices = list(range(densitytensor.ndim))
    for i in indices_to_contract:
        einsum_indices[i + n_qubits] = einsum_indices[i]
    densitytensor = jnp.einsum(densitytensor, einsum_indices)
    return densitytensor

def get_params_to_densitytensor_func(gate_seq: Sequence[Union[str,
                                                              jnp.ndarray,
                                                              Callable[[jnp.ndarray], jnp.ndarray],
                                                              Callable[[], jnp.ndarray]]],
                                     qubit_inds_seq: Sequence[Sequence[int]],
                                     param_inds_seq: Sequence[Sequence[int]],
                                     n_qubits: int = None) -> UnionCallableOptionalArray:
    """
    Creates a function that maps circuit parameters to a density tensor.
    densitytensor = densitymatrix.reshape((2,) * 2 * n_qubits)
    densitymatrix = densitytensor.reshape(2 ** n_qubits, 2 ** n_qubits)

    Args:
        gate_seq: Sequence of gates.
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

    additional_operations=("discard", "create")
    check_circuit(gate_seq,
                  qubit_inds_seq,
                  param_inds_seq,
                  n_qubits,
                  additional_operations=additional_operations)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    gate_seq_callable = _to_gate_funcs(gate_seq, ignore=additional_operations)
    param_inds_seq = _arrayify_inds(param_inds_seq)


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
        nonlocal n_qubits
        # Keeps track of created/discarded qubit indices
        qubit_dict = {i:i for i in range(n_qubits)}
        if densitytensor_in is None:
            densitytensor = jnp.zeros((2,) * 2 * n_qubits)
            densitytensor = densitytensor.at[(0,) * 2 * n_qubits].set(1.)
        else:
            densitytensor = densitytensor_in
        params = jnp.atleast_1d(params)
        for operation, qubit_inds, param_inds in zip(gate_seq_callable, qubit_inds_seq, param_inds_seq):
            if operation == "discard":
                nonexistant_qubits = set(qubit_inds) - set(qubit_dict.keys())
                if nonexistant_qubits:
                    raise ValueError(f"Trying to discard nonexistant qubits {nonexistant_qubits}")
                indices_to_delete = [qubit_dict[i] for i in qubit_inds]
                densitytensor = _partial_trace(densitytensor, indices_to_delete)
                for i in qubit_inds:
                    # Delete discarded qubit indices
                    del qubit_dict[i]
                    # Track shift of indices greater than discarded ones
                    for k in qubit_dict:
                        if qubit_dict[k] > i:
                            qubit_dict[k] -= 1
            elif operation == "create":
                if set(qubit_inds) & set(qubit_dict.keys()):
                    raise ValueError("Trying to create existing qubits")
                if len(param_inds) > 1:
                    raise ValueError("Multiple density tensors not yet supported when creating qubits")
                new_qubit_dt = params[param_inds[0]]
                if len(qubit_inds) is not new_qubit_dt.ndim // 2:
                    raise ValueError(f"Trying to create {len(qubit_inds)} qubits using density tensor representing {new_qubit_dt.ndim // 2} qubits")

                original_nr_qubits = densitytensor.ndim // 2

                # Tensor deisity matrices ensuring co- and contravariant indices are correctly ordered
                indices_1 = jnp.array(range(densitytensor.ndim))
                indices_2 = jnp.array(range(densitytensor.ndim, densitytensor.ndim + new_qubit_dt.ndim))
                indices_out = jnp.c_[indices_1.reshape(2, -1), indices_2.reshape(2, -1)].reshape(-1)
                densitytensor = jnp.einsum(densitytensor, indices_1.tolist(),
                                           new_qubit_dt, indices_2.tolist(),
                                           indices_out.tolist())

                # Keep track of new qubit indices
                for n, i in enumerate(qubit_inds):
                    qubit_dict[i] = original_nr_qubits + n
            else:
                qubit_inds = [qubit_dict[i] for i in qubit_inds]
                gate_params = jnp.take(params, param_inds)
                gate_unitary = operation(*gate_params)
                gate_unitary = gate_unitary.reshape((2,) * (2 * len(qubit_inds)))  # Ensure gate is in tensor form
                densitytensor = _kraus_single(densitytensor, gate_unitary, qubit_inds)
        return densitytensor

    if all([pi.size == 0 for pi in param_inds_seq]):
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
