from jax import numpy as jnp

_I = jnp.eye(2)

_X = jnp.array([[0., 1.],
                [1., 0.]])

_Y = jnp.array([[0., -1.j],
                [1.j, 0.]])

_Z = jnp.array([[1., 0.],
                [0., -1.]])

_H = jnp.array([[1., 1.],
                [1., -1]]) / jnp.sqrt(2)

_S = jnp.array([[1., 0.],
                [0., jnp.exp(1.j)]])

_T = jnp.array([[1., 0.],
                [0., jnp.exp(jnp.pi * 1.j / 4)]])

_CX = jnp.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 0., 1.],
                 [0., 0., 1., 0.]]).reshape((2,) * 4)

_CY = jnp.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 0., -1.j],
                 [0., 0., 1.j, 0.]]).reshape((2,) * 4)

_CZ = jnp.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., -1.]]).reshape((2,) * 4)


def X() -> jnp.ndarray:
    return _X


def Y() -> jnp.ndarray:
    return _Y


def Z() -> jnp.ndarray:
    return _Z


def H() -> jnp.ndarray:
    return _H


def S() -> jnp.ndarray:
    return _S


def T() -> jnp.ndarray:
    return _T


def CX() -> jnp.ndarray:
    return _CX


def CY() -> jnp.ndarray:
    return _CY


def CZ() -> jnp.ndarray:
    return _CZ


def Rx(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * _I - jnp.sin(param_pi_2) * _X * 1.j


def Ry(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * _I - jnp.sin(param_pi_2) * _Y * 1.j


def Rz(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * _I - jnp.sin(param_pi_2) * _Z * 1.j
