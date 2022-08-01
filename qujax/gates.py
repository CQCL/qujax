from jax import numpy as jnp

I = jnp.eye(2)

X = jnp.array([[0., 1.],
               [1., 0.]])

Y = jnp.array([[0., -1.j],
               [1.j, 0.]])

Z = jnp.array([[1., 0.],
               [0., -1.]])

H = jnp.array([[1., 1.],
               [1., -1]]) / jnp.sqrt(2)

S = jnp.array([[1., 0.],
               [0., 1.j]])

Sdg = jnp.array([[1., 0.],
                 [0., -1.j]])

T = jnp.array([[1., 0.],
               [0., jnp.exp(jnp.pi * 1.j / 4)]])

Tdg = jnp.array([[1., 0.],
                 [0., jnp.exp(-jnp.pi * 1.j / 4)]])

V = jnp.array([[1., -1.j],
               [-1.j, 1.]]) / jnp.sqrt(2)

Vdg = jnp.array([[1., 1.j],
                 [1.j, 1.]]) / jnp.sqrt(2)

SX = jnp.array([[1. + 1.j, 1. - 1.j],
                [1. - 1.j, 1. + 1.j]]) / 2

SXdg = jnp.array([[1. - 1.j, 1. + 1.j],
                  [1. + 1.j, 1. - 1.j]]) / 2

CX = jnp.array([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [0., 0., 1., 0.]]).reshape((2,) * 4)

CY = jnp.array([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., -1.j],
                [0., 0., 1.j, 0.]]).reshape((2,) * 4)

CZ = jnp.array([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., -1.]]).reshape((2,) * 4)


def Rx(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * X * 1.j


def Ry(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * Y * 1.j


def Rz(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * Z * 1.j


def U1(param: float) -> jnp.ndarray:
    return U3(0, 0, param)


def U2(param1: float, param2: float) -> jnp.ndarray:
    return U3(0.5, param1, param2)


def U3(param1: float, param2: float, param3: float) -> jnp.ndarray:
    return jnp.exp((param2 + param3) * jnp.pi * 1.j / 2) * Rz(param2) @ Ry(param1) @ Rz(param3)

