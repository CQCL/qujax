import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from qujax.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'qujax'
copyright = '2022, Sam Duffield'
author = 'Sam Duffield'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

autodoc_typehints = "description"

html4_writer = True

autodoc_type_aliases = {
    'jnp.ndarray': 'ndarray',
    'random.PRNGKeyArray': 'jax.random.PRNGKeyArray',
    'UnionCallableOptionalArray': 'Union[Callable[[ndarray, Optional[ndarray]], ndarray], '
                                  'Callable[[Optional[ndarray]], ndarray]]'

}

latex_engine = 'pdflatex'
