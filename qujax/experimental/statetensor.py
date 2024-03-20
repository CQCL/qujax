from typing import Any, Callable, Sequence, Tuple, Optional, Mapping, Union

# Backwards compatibility with Python <3.10
from typing_extensions import TypeVarTuple, Unpack

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qujax import gates
from qujax.typing import Gate, GateFunction

from qujax.statetensor import apply_gate

PyTree = Any

Operation = Union[
    Gate,
    str,
]


def wrap_parameterised_tensor(
    gate_func: Callable, qubit_inds: Sequence[int]
) -> Callable:
    """
    Takes a callable representing a parameterised gate and wraps it in a function that takes
    the returned jax.Array and applies it to the qubits specified by `qubit_inds`.

    Args:
        gate_func: Callable representing parameterised gate.
        qubit_inds: Indices gate is to be applied to.

    Returns:
        Callable taking in gate parameters, input statetensor and input classical registers,
        and returning updated statetensor after applying parameterized gate to specified qubits.
    """

    def unitary_op(
        params: Tuple[jax.Array],
        statetensor_in: jax.Array,
        classical_registers_in: PyTree,
    ):
        gate_unitary = gate_func(*params[0])
        statetensor = apply_gate(statetensor_in, gate_unitary, qubit_inds)

        return statetensor, classical_registers_in

    return unitary_op


def _array_to_callable(arr: jax.Array) -> Callable[[], jax.Array]:
    """
    Wraps array `arr` in a callable that takes no parameters and returns `arr`.
    """

    def _no_param_tensor():
        return arr

    return _no_param_tensor


def _to_gate_func(
    gate: Gate,
    tensor_dict: Mapping[str, Union[Callable, jax.Array]],
) -> GateFunction:
    """
    Converts a gate specification to a callable function that takes the gate parameters and returns
    the corresponding unitary.

    Args:
        gate: Gate specification. Can be either a string, a callable or a jax.Array.

    Returns:
        Callable taking gate parameters and returning
    """

    if isinstance(gate, str):
        gate = tensor_dict[gate]
    if isinstance(gate, jax.Array):
        gate = _array_to_callable(gate)
    if callable(gate):
        return gate
    else:
        raise TypeError(
            f"Unsupported gate type - gate must be either a string in qujax.gates, a jax.Array or "
            f"callable: {type(gate)}"
        )


def parse_op(
    op: Operation,
    params: Sequence[Any],
    gate_dict: Mapping[str, Union[Callable, jax.Array]],
    op_dict: Mapping[str, Callable],
) -> Callable:
    """
    Parses operation specified by `op`, applying relevant metaparameters and returning a callable
    retpresenting the operation to be applied to the circuit.

    Args:
        op: Operation specification. Can be:
            - A string, in which case we first check whether it is a gate by looking it up in
            `tensor_dict` and then check whether it is a more general operation by looking it up
            in `op_dict`.
            - A jax.Array, which we assume to represent a gate.
            - A callable, which we assume to represent a parameterized gate.
        params: Operator metaparameters. For gates, these are the qubit indices the gate is to
            be applied to.
        tensor_dict: Dictionary mapping strings to gates.
        op_dict: Dictionary mapping strings to callables that take operation metaparameters and
            return a function representing the operation to be applied to the circuit.

    Returns:
        A callable encoding the operation to be applied to the circuit.
    """
    # Gates
    if (
        (isinstance(op, str) and op in gate_dict)
        or isinstance(op, jax.Array)
        or callable(op)
    ):
        op = _to_gate_func(op, gate_dict)
        return wrap_parameterised_tensor(op, params)

    if isinstance(op, str) and op in op_dict:
        return op_dict[op](*params)

    if isinstance(op, str):
        raise ValueError(f"String {op} not a known gate or operation")
    else:
        raise TypeError(
            f"Invalid specification for `op`, got type {type(op)} with value {op}"
        )


def get_default_gates() -> dict:
    """
    Returns dictionary of default gates supported by qujax.
    """
    return {
        k: v for k, v in gates.__dict__.items() if not k.startswith(("_", "jax", "jnp"))
    }


def _gate_func_to_unitary(
    gate_func: GateFunction,
    n_qubits: int,
    params: jax.Array,
) -> jax.Array:
    """
    Compute tensor representing parameterised unitary for specific parameters.

    Args:
        gate_func: Function that maps a (possibly empty) parameter array to a unitary tensor
        n_qubts: Number of qubits unitary acts on
        params: Parameter vector

    Returns:
        Array containing gate unitary in tensor form.
    """
    gate_unitary = gate_func(*params)
    gate_unitary = gate_unitary.reshape(
        (2,) * (2 * n_qubits)
    )  # Ensure gate is in tensor form
    return gate_unitary


Op = Callable[
    [Tuple[jax.Array, ...], jax.Array, jax.Array], Tuple[jax.Array, jax.Array]
]
OpSpecArgs = TypeVarTuple("OpSpecArgs")
OpSpec = Callable[[Unpack[OpSpecArgs]], Op]


def get_default_operations(
    gate_dict: Mapping[str, Union[Callable, jax.Array]]
) -> Mapping[str, OpSpec]:
    """
    Returns dictionary of default operations supported by qujax. Each operation is a function
    that takes a set of metaparemeters and returns another function. The returned function
    must have three arguments: `op_params`, `statetensor_in` and `classical_registers_in`.
    `op_params` holds parameters that are passed when the circuit is executed, while
    `statetensor_in` and `classical_registers_in` correspond to the statetensor
    and classical registers, respectively, being modified by the circuit.

    Parameters:
        `gate_dict`: Dictionary encoding quantum gates that the circuit can use. This
            dictionary maps strings to a callable in the case of parameterized gates or to a
            jax.Array in the case of unparameterized gates.
    """
    op_dict: dict[str, OpSpec] = dict()

    def generic_op(f: Op) -> Op:
        """
        Generic operation to be applied to the circuit, passed as a metaparameter `f`.
        """
        return f

    def conditional_gate(gates: Sequence[Gate], qubit_inds: Sequence[int]) -> Op:
        """
        Operation applying one of the gates in `gates` according to an index passed as a
        circuit parameter.

        Args:
            gates: gates from which one is selected to be applied
            qubit_indices: indices of qubits the selected gate is to be applied to
        """
        gate_funcs = [_to_gate_func(g, gate_dict) for g in gates]

        def apply_conditional_gate(
            op_params: Union[Tuple[jax.Array], Tuple[jax.Array, jax.Array]],
            statetensor_in: jax.Array,
            classical_registers_in: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            """
            Applies a gate specified by an index passed in `op_params` to a statetensor.

            Args:
                op_params: gates from which one is selected to be applied
                statetensor_in: indices of qubits the selected gate is to be applied to
                classical_registers_in: indices of qubits the selected gate is to be applied to
            """
            if len(op_params) == 1:
                ind, gate_params = op_params[0], jnp.empty((len(gates), 0))
            elif len(op_params) == 2:
                ind, gate_params = op_params[0], jnp.array(op_params[1])
            else:
                raise ValueError("Invalid number of parameters for ConditionalGate")

            unitaries = jnp.stack(
                [
                    _gate_func_to_unitary(
                        gate_funcs[i], len(qubit_inds), gate_params[i]
                    )
                    for i in range(len(gate_funcs))
                ]
            )

            chosen_unitary = unitaries[ind]

            statevector = apply_gate(statetensor_in, chosen_unitary, qubit_inds)
            return statevector, classical_registers_in

        return apply_conditional_gate

    op_dict["Generic"] = generic_op
    op_dict["ConditionalGate"] = conditional_gate

    return op_dict


ParamInds = Optional[
    Union[
        int,
        Sequence[int],
        Sequence[Sequence[int]],
        Mapping[str, int],
        Mapping[str, Sequence[int]],
    ]
]


def get_params(
    param_inds: ParamInds,
    params: Union[Mapping[str, ArrayLike], ArrayLike],
) -> Tuple[Any, ...]:
    """
    Extracts parameters from `params` using indices specified by `param_inds`.

    Args:
        param_inds: Indices of parameters. Can be
            - None (results in an empty jax.Array)
            - an integer, when `params` is an indexable array
            - a dictionary, when `params` is also a dictionary
            - nested list or tuples of the above
        params: Parameters from which a subset is picked. Can be either an array or a dictionary
            of arrays
    Returns:
        Tuple of indexed parameters respeciting the structure of nested lists/tuples of param_inds.

    """
    op_params: Tuple[Any, ...]
    if param_inds is None:
        op_params = (jnp.array([]),)
    elif isinstance(param_inds, int) and isinstance(params, jax.Array):
        op_params = (params[param_inds],)
    elif isinstance(param_inds, dict) and isinstance(params, dict):
        op_params = tuple(
            jnp.take(params[k], jnp.array(param_inds[k])) for k in param_inds
        )
    elif isinstance(param_inds, (list, tuple)):
        if len(param_inds):
            if all(isinstance(x, int) for x in param_inds):
                op_params = (jnp.take(params, jnp.array(param_inds)),)
            else:
                op_params = tuple(get_params(p, params) for p in param_inds)
        else:
            op_params = (jnp.array([]),)
    else:
        raise TypeError(
            f"Invalid specification for parameters: {type(param_inds)=} {type(params)=}."
        )
    return op_params


def get_params_to_statetensor_func(
    op_seq: Sequence[Operation],
    op_metaparams_seq: Sequence[Sequence[Any]],
    param_pos_seq: Sequence[ParamInds],
    op_dict: Optional[Mapping[str, OpSpec]] = None,
    gate_dict: Optional[Mapping[str, Union[jax.Array, GateFunction]]] = None,
):
    """
    Creates a function that maps circuit parameters to a statetensor.

    Args:
        op_seq: Sequence of operations to be executed.
            Can be either
            - a string specifying a gate in `gate_dict`
            - a jax.Array specifying a gate
            - a function returning a jax.Array specifying a parameterized gate.
            - a string specifying an operation in `op_dict`
        op_params_seq: Sequence of operation meta-parameters. Each element corresponds to one
            operation in `op_seq`. For gates, this will be the qubit indices the gate is applied to.
        param_pos_seq: Sequence of indices specifying the positions of the parameters each gate
            or operation takes.
            Note that these are parameters of the circuit, and are distinct from the meta-parameters
            fixed in `op_params_seq`.
        op_dict: Dictionary mapping strings to operations. Each operation is a function
            taking metaparameters (which are specified in `op_params_seq`) and returning another
            function. This returned function encodes the operation, and takes an array of
            parameters, a statetensor and classical registers, and returns the updated statetensor
            and classical registers after the operation is applied.
        gate_dict: Dictionary mapping strings to gates. Each gate is either a jax.Array or a
            function taking a number of parameters and returning a jax.Array.
            Defaults to qujax's dictionary of gates.
    Returns:
        Function that takes a number of parameters, an input statetensor and an input set of
        classical registers, and returns the updated statetensor and classical registers
        after the specified gates and operations are applied.
    """
    if gate_dict is None:
        gate_dict = get_default_gates()
    if op_dict is None:
        op_dict = get_default_operations(gate_dict)

    repeated_ops = set(gate_dict.keys()) & set(op_dict.keys())
    if repeated_ops:
        raise ValueError(
            f"Operator list and gate list have repeated operation(s): {repeated_ops}"
        )

    parsed_op_seq = [
        parse_op(op, params, gate_dict, op_dict)
        for op, params in zip(op_seq, op_metaparams_seq)
    ]

    def params_to_statetensor_func(
        params: Union[Mapping[str, ArrayLike], ArrayLike],
        statetensor_in: jax.Array,
        classical_registers_in: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, PyTree]:
        """
        Applies parameterised circuit to the quantum state represented by `statetensor_in`.

        Args:
            params: Parameters to be passed to the circuit
            statetensor_in: Input state in tensor form.
            classical_registers_in: Classical registers that can store intermediate results
                (e.g. measurements), possibly to later reuse them
        Returns:
            Resulting quantum state and classical registers after applying the circuit.

        """
        statetensor = statetensor_in
        classical_registers = classical_registers_in
        for (
            op,
            param_pos,
        ) in zip(
            parsed_op_seq,
            param_pos_seq,
        ):
            op_params = get_params(param_pos, params)
            statetensor, classical_registers = op(
                op_params, statetensor, classical_registers_in
            )

        return statetensor, classical_registers

    return params_to_statetensor_func
