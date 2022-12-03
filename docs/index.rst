
Welcome to qujax's documentation!
=================================

``qujax`` simulates quantum computation using `JAX <https://github.com/google/jax>`_.

Source code can be found on `GitHub <https://github.com/CQCL/qujax>`_, including a suite of `example notebooks <https://github.com/CQCL/qujax/tree/main/examples>`_.

The `pytket-qujax <https://github.com/CQCL/pytket-qujax>`_, extension is useful for converting a `tket <https://github.com/CQCL/pytket>`_ circuit into ``qujax``.

Note that ``qujax`` assumes parameters are given in units of π (i.e. in [0,2] rather than [0, 2π]).

Install
=================================
``qujax`` is hosted on `PyPI (aka the Cheese Shop) <https://pypi.org/project/qujax/>`_ and can be installed with

``pip install qujax``


Docs
=================================

.. toctree::

    apply_gate
    get_params_to_statetensor_func
    get_params_to_unitarytensor_func
    get_statetensor_to_expectation_func
    get_statetensor_to_sampled_expectation_func
    integers_to_bitstrings
    bitstrings_to_integers
    sample_integers
    sample_bitstrings
    check_circuit
    print_circuit
    densitytensor
    gates <https://github.com/CQCL/qujax/blob/main/qujax/gates.py>


.. toctree::
    :caption: Links:
    :hidden:

    GitHub <https://github.com/CQCL/qujax>
    Example notebooks <https://github.com/CQCL/qujax/tree/main/examples>
    PyPI <https://pypi.org/project/qujax/>
    pytket-qujax <https://cqcl.github.io/pytket-qujax/api>




