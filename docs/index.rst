
Welcome to qujax's documentation!
=================================

``qujax`` is a `JAX <https://github.com/google/jax>`_-based Python library for the classical simulation of quantum circuits. It is designed to be *simple*, *fast* and *flexible*.


It follows a functional programming design by translating circuits into pure functions. This allows qujax to `seamlessly and directly interface with JAX <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_, enabling direct access to its powerful automatic differentiation tools, just-in-time compiler, vectorization capabilities, GPU/TPU integration and growing ecosystem of packages.

If you are new to the library, we recommend that you head to the :doc:`getting_started` section of the documentation. More advanced use-cases, including the training of parameterised quantum circuits, can be found in :doc:`examples`.

The source code can be found on `GitHub <https://github.com/CQCL/qujax>`_. The `pytket-qujax <https://github.com/CQCL/pytket-qujax>`_ extension can be used to translate a `tket <https://github.com/CQCL/pytket>`_ circuit directly into ``qujax``.

**Important note**: qujax circuit parameters are expressed in units of :math:`\pi` (e.g. in the range :math:`[0,2]` as opposed to :math:`[0, 2\pi]`).

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
   :caption: Documentation:
   :titlesonly:
    
    Getting started <getting_started>
    Examples <examples>
    List of gates <gates>

.. toctree::
   :caption: API Reference:
   :titlesonly:
   :maxdepth: 1

    Pure state simulation <statetensor>
    Mixed state simulation <densitytensor>
    Utility functions <utils>
    Experimental <experimental>

.. toctree::
    :caption: Links:
    :hidden:

    GitHub <https://github.com/CQCL/qujax>
    Paper <https://doi.org/10.21105/joss.05504>
    PyPI <https://pypi.org/project/qujax/>
    pytket-qujax <https://cqcl.github.io/pytket-qujax/api>
