from jax import numpy as jnp

I = jnp.eye(2)

_0 = jnp.zeros((2, 2))

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

CX = jnp.block([[I, _0],
                [_0, X]]).reshape((2,) * 4)

CY = jnp.block([[I, _0],
                [_0, Y]]).reshape((2,) * 4)

CZ = jnp.block([[I, _0],
                [_0, Z]]).reshape((2,) * 4)

CH = jnp.block([[I, _0],
                [_0, H]]).reshape((2,) * 4)

CV = jnp.block([[I, _0],
                [_0, V]]).reshape((2,) * 4)

CVdg = jnp.block([[I, _0],
                  [_0, Vdg]]).reshape((2,) * 4)

CSX = jnp.block([[I, _0],
                 [_0, SX]]).reshape((2,) * 4)

CSXdg = jnp.block([[I, _0],
                   [_0, SXdg]]).reshape((2,) * 4)

CCX = jnp.block([[I, _0, _0, _0],  # Toffoli gate
                 [_0, I, _0, _0],
                 [_0, _0, I, _0],
                 [_0, _0, _0, X]]).reshape((2,) * 6)

ECR = jnp.block([[_0, Vdg],
                 [V, _0]]).reshape((2,) * 4)

SWAP = jnp.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., 1]])

CSWAP = jnp.block([[jnp.eye(4), jnp.zeros((4, 4))],
                   [jnp.zeros((4, 4)), SWAP]]).reshape((2,) * 6)


def Rx(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * X * 1.j


def Ry(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * Y * 1.j


def Rz(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    return jnp.cos(param_pi_2) * I - jnp.sin(param_pi_2) * Z * 1.j


def CRx(param: float) -> jnp.ndarray:
    return jnp.block([[I, _0],
                      [_0, Rx(param)]]).reshape((2,) * 4)


def CRy(param: float) -> jnp.ndarray:
    return jnp.block([[I, _0],
                      [_0, Ry(param)]]).reshape((2,) * 4)


def CRz(param: float) -> jnp.ndarray:
    return jnp.block([[I, _0],
                      [_0, Rz(param)]]).reshape((2,) * 4)


def U1(param: float) -> jnp.ndarray:
    return U3(0, 0, param)


def U2(param0: float, param1: float) -> jnp.ndarray:
    return U3(0.5, param0, param1)


def U3(param0: float, param1: float, param2: float) -> jnp.ndarray:
    return jnp.exp((param1 + param2) * jnp.pi * 1.j / 2) * Rz(param1) @ Ry(param0) @ Rz(param2)


def CU1(param: float) -> jnp.ndarray:
    return jnp.block([[I, _0],
                      [_0, U1(param)]]).reshape((2,) * 4)


def CU2(param0: float, param1: float) -> jnp.ndarray:
    return jnp.block([[I, _0],
                      [_0, U2(param0, param1)]]).reshape((2,) * 4)


def CU3(param0: float, param1: float, param2: float) -> jnp.ndarray:
    return jnp.block([[I, _0],
                      [_0, U3(param0, param1, param2)]]).reshape((2,) * 4)


def ISWAP(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    c = jnp.cos(param_pi_2)
    i_s = 1.j * jnp.sin(param_pi_2)
    return jnp.array([[1., 0., 0., 0.],
                      [0., c, i_s, 0.],
                      [0., i_s, c, 0.],
                      [0., 0., 0., 1.]]).reshape((2,) * 4)


def PhasedISWAP(param0: float, param1: float) -> jnp.ndarray:
    param1_pi_2 = param1 * jnp.pi / 2
    c = jnp.cos(param1_pi_2)
    i_s = 1.j * jnp.sin(param1_pi_2)
    return jnp.array([[1., 0., 0., 0.],
                      [0., c, i_s * jnp.exp(2.j * jnp.pi * param0), 0.],
                      [0., i_s * jnp.exp(-2.j * jnp.pi * param0), c, 0.],
                      [0., 0., 0., 1.]]).reshape((2,) * 4)


def XXPhase(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    c = jnp.cos(param_pi_2)
    i_s = 1.j * jnp.sin(param_pi_2)
    return jnp.array([[c, 0., 0., -i_s],
                      [0., c, -i_s, 0.],
                      [0., -i_s, c, 0.],
                      [-i_s, 0., 0., c]]).reshape((2,) * 4)


def YYPhase(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    c = jnp.cos(param_pi_2)
    i_s = 1.j * jnp.sin(param_pi_2)
    return jnp.array([[c, 0., 0., i_s],
                      [0., c, -i_s, 0.],
                      [0., -i_s, c, 0.],
                      [i_s, 0., 0., c]]).reshape((2,) * 4)


def ZZPhase(param: float) -> jnp.ndarray:
    param_pi_2 = param * jnp.pi / 2
    e_m = jnp.exp(-1.j * param_pi_2)
    e_p = jnp.exp(1.j * param_pi_2)
    return jnp.diag(jnp.array([e_m, e_p, e_p, e_m])).reshape((2,) * 4)
