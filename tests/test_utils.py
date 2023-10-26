import jax
import jax.numpy as jnp

import qujax


def test_repeat_circuit():
    n_qubits = 4
    depth = 4
    seed = 0
    statetensor_in = qujax.all_zeros_statetensor(n_qubits)
    densitytensor_in = qujax.all_zeros_densitytensor(n_qubits)

    def circuit(n_qubits: int, depth: int):
        parameter_index = 0

        gates = []
        qubit_inds = []
        param_inds = []

        for _ in range(depth):
            # Rx layer
            for i in range(n_qubits):
                gates.append("Rx")
                qubit_inds.append([i])
                param_inds.append([parameter_index])
                parameter_index += 1

            # CRz layer
            for i in range(n_qubits - 1):
                gates.append("CRz")
                qubit_inds.append([i, i + 1])
                param_inds.append([parameter_index])
                parameter_index += 1

        return gates, qubit_inds, param_inds, parameter_index

    rng = jax.random.PRNGKey(seed)

    g1, qi1, pi1, np1 = circuit(n_qubits, depth)

    params = jax.random.uniform(rng, (np1,))

    param_to_st = qujax.get_params_to_statetensor_func(g1, qi1, pi1, n_qubits)

    g2, qi2, pi2, np2 = circuit(n_qubits, 1)

    param_to_st_single_repetition = qujax.get_params_to_statetensor_func(
        g2, qi2, pi2, n_qubits
    )
    param_to_st_repeated = qujax.repeat_circuit(param_to_st_single_repetition, np2)

    assert jnp.allclose(
        param_to_st(params, statetensor_in),
        param_to_st_repeated(params, statetensor_in),
    )

    param_to_dt = qujax.get_params_to_densitytensor_func(g1, qi1, pi1, n_qubits)

    param_to_dt_single_repetition = qujax.get_params_to_densitytensor_func(
        g2, qi2, pi2, n_qubits
    )
    param_to_dt_repeated = qujax.repeat_circuit(param_to_dt_single_repetition, np2)

    assert jnp.allclose(
        param_to_dt(params, densitytensor_in),
        param_to_dt_repeated(params, densitytensor_in),
    )
