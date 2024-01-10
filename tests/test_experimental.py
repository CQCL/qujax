import jax
import jax.numpy as jnp

import qujax
from qujax import all_zeros_statetensor, apply_gate
from qujax.experimental.statetensor import get_params_to_statetensor_func


def test_get_params_to_statetensor_func():
    ops = ["H", "H", "H", "CX", "Rz", "CY"]
    op_params = [[0], [1], [2], [0, 1], [1], [1, 2]]
    param_inds = [[], [], [], None, [0], []]

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)
    param_to_st = jax.jit(param_to_st)
    param = jnp.array(0.1)
    st_in = all_zeros_statetensor(3)
    st, _ = param_to_st(param, st_in)

    true_sv = jnp.array(
        [
            0.34920055 - 0.05530793j,
            0.34920055 - 0.05530793j,
            0.05530793 - 0.34920055j,
            -0.05530793 + 0.34920055j,
            0.34920055 - 0.05530793j,
            0.34920055 - 0.05530793j,
            0.05530793 - 0.34920055j,
            -0.05530793 + 0.34920055j,
        ],
        dtype="complex64",
    )

    assert st.size == true_sv.size
    assert jnp.allclose(st.flatten(), true_sv)


def test_stochasticity():
    ops = ["ConditionalGate"]
    op_params = [[["X", "Y", "Z"], [0]]]
    param_inds = [[{"op_ind": 0}]]

    st_in = all_zeros_statetensor(1)
    X_apply = apply_gate(st_in, qujax.gates.X, [0])
    Y_apply = apply_gate(st_in, qujax.gates.Y, [0])
    Z_apply = apply_gate(st_in, qujax.gates.Z, [0])

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)
    param_to_st = jax.jit(param_to_st)

    st_in = all_zeros_statetensor(1)

    st_X, _ = param_to_st({"op_ind": 0}, st_in)
    st_Y, _ = param_to_st({"op_ind": 1}, st_in)
    st_Z, _ = param_to_st({"op_ind": 2}, st_in)

    assert jnp.allclose(X_apply, st_X)
    assert jnp.allclose(Y_apply, st_Y)
    assert jnp.allclose(Z_apply, st_Z)


def test_parameterised_stochasticity():
    ops = ["ConditionalGate"]
    op_params = [[["Rx", "Ry", "Rz"], [0]]]
    param_inds = [[{"op_ind": 0}, [{"angles": 0}, {"angles": 1}, {"angles": 2}]]]

    st_in = all_zeros_statetensor(1)
    params = jnp.array([0.1, 0.2, 0.3])

    CX_apply = apply_gate(st_in, qujax.gates.Rx(params[0].item()), [0])
    CY_apply = apply_gate(st_in, qujax.gates.Ry(params[1].item()), [0])
    CZ_apply = apply_gate(st_in, qujax.gates.Rz(params[2].item()), [0])

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)

    st_in = all_zeros_statetensor(1)

    st_CX, _ = param_to_st({"angles": params, "op_ind": 0}, st_in)
    st_CY, _ = param_to_st({"angles": params, "op_ind": 1}, st_in)
    st_CZ, _ = param_to_st({"angles": params, "op_ind": 2}, st_in)

    assert jnp.allclose(CX_apply, st_CX)
    assert jnp.allclose(CY_apply, st_CY)
    assert jnp.allclose(CZ_apply, st_CZ)

    batched_op_inds = jnp.array([[0], [1], [2]])

    batched_param_to_st = jax.vmap(
        param_to_st, in_axes=({"angles": None, "op_ind": 0}, None)
    )

    batched_st, _ = batched_param_to_st(
        {"angles": params, "op_ind": batched_op_inds}, st_in
    )

    assert jnp.allclose(batched_st, jnp.stack([st_CX, st_CY, st_CZ]))
