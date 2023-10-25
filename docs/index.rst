
Welcome to qujax's documentation!
=================================

``qujax`` is a `JAX <https://github.com/google/jax>`_-based Python library for the classical simulation of quantum circuits. It is designed to be *simple*, *fast* and *flexible*.

It follows a functional programming design by translating circuits into pure functions. This allows qujax to `seamlessly and directly interface with JAX <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_.

Source code can be found on `GitHub <https://github.com/CQCL/qujax>`_, including a suite of `example notebooks <https://github.com/CQCL/qujax#examples>`_.

The `pytket-qujax <https://github.com/CQCL/pytket-qujax>`_ extension can be used to translate a `tket <https://github.com/CQCL/pytket>`_ circuit directly into ``qujax``.

**Note that ``qujax`` assumes parameters are given in units of π (i.e. in [0,2] rather than [0, 2π]).**

Install
=================================
``qujax`` is hosted on `PyPI <https://pypi.org/project/qujax/>`_ and can be installed with

.. code-block:: bash

   pip install qujax

Cite
=================================
If you have used qujax in your code or research, we kindly ask that you cite it. You can use the following BibTeX entry for this:

.. code-block:: bibtex

   @article{qujax2023,
     author = {Duffield, Samuel and Matos, Gabriel and Johannsen, Melf},
     doi = {10.21105/joss.05504},
     journal = {Journal of Open Source Software},
     month = sep,
     number = {89},
     pages = {5504},
     title = {{qujax: Simulating quantum circuits with JAX}},
     url = {https://joss.theoj.org/papers/10.21105/joss.05504},
     volume = {8},
     year = {2023}
   }

Contents
=================================

.. toctree::
   :caption: API Reference:

    Pure state simulation <statetensor>
    Mixed state simulation <densitytensor>
    Utility functions <utils>
    List of gates <https://github.com/CQCL/qujax/blob/main/qujax/gates.py>

.. toctree::
    :caption: Links:
    :hidden:

    GitHub <https://github.com/CQCL/qujax>
    Paper <https://doi.org/10.21105/joss.05504>
    Example notebooks <https://github.com/CQCL/qujax#examples>
    PyPI <https://pypi.org/project/qujax/>
    pytket-qujax <https://cqcl.github.io/pytket-qujax/api>
