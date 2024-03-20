from typing import Union, Optional, Protocol, Callable, Sequence

# Backwards compatibility with Python <3.10
from typing_extensions import TypeVarTuple, Unpack

import jax
from jax.typing import ArrayLike


class PureParameterizedCircuit(Protocol):
    def __call__(
        self, params: ArrayLike, statetensor_in: Optional[jax.Array] = None
    ) -> jax.Array:
        ...


class PureUnparameterizedCircuit(Protocol):
    def __call__(self, statetensor_in: Optional[jax.Array] = None) -> jax.Array:
        ...


class MixedParameterizedCircuit(Protocol):
    def __call__(
        self, params: ArrayLike, densitytensor_in: Optional[jax.Array] = None
    ) -> jax.Array:
        ...


class MixedUnparameterizedCircuit(Protocol):
    def __call__(self, densitytensor_in: Optional[jax.Array] = None) -> jax.Array:
        ...


GateArgs = TypeVarTuple("GateArgs")
# Function that takes arbitrary nr. of parameters and returns an array representing the gate
# Currently Python does not allow us to restrict the type of the arguments using a TypeVarTuple
ParameterizedGateFunction = Callable[[Unpack[GateArgs]], jax.Array]
UnparameterizedGateFunction = Callable[[], jax.Array]
GateFunction = Union[ParameterizedGateFunction, UnparameterizedGateFunction]
GateParameterIndices = Optional[Sequence[int]]

PureCircuitFunction = Union[PureUnparameterizedCircuit, PureParameterizedCircuit]
MixedCircuitFunction = Union[MixedUnparameterizedCircuit, MixedParameterizedCircuit]

Gate = Union[str, jax.Array, GateFunction]

KrausOp = Union[Gate, Sequence[Gate]]
