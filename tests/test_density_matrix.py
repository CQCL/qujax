from jax import numpy as jnp, jit

import qujax
from qujax.density_matrix import _kraus_single


def test_kraus_single():
    n_qubits = 3
    dim = 2 ** n_qubits
    density_matrix = jnp.arange(dim**2).reshape(dim, dim)
    density_tensor = density_matrix.reshape((2,) * 2 * n_qubits)
    kraus_operator = qujax.gates.Rx(0.2)

    qubit_inds = (1,)

    qujax_kraus_dt = _kraus_single(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)

    unitary_matrix = jnp.kron(jnp.eye(2 * qubit_inds[0]), kraus_operator)
    unitary_matrix = jnp.kron(unitary_matrix, jnp.eye(2 * (n_qubits - qubit_inds[0] - 1)))
    check_kraus_dm = unitary_matrix @ density_matrix @ unitary_matrix.conj().T

    assert jnp.all(jnp.abs(qujax_kraus_dm - check_kraus_dm) < 1e-5)

    qujax_kraus_dt_jit = jit(_kraus_single, static_argnums=(2,))(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.all(jnp.abs(qujax_kraus_dm_jit - check_kraus_dm) < 1e-5)

