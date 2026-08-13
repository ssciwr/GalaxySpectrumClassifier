# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from GalaxySpectrumClassifier import __version__

# -- Project information -----------------------------------------------------

project = "GalaxySpectrumClassifier"
copyright = "2026, Harald Mack"
author = "Harald Mack"

# The package is installed into the documentation environment, so the version
# is taken from there rather than being maintained separately.
release = __version__
version = release.split("+")[0]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_mdinclude",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# "build" is the output directory used by the Makefile, which lives inside
# this source directory.
exclude_patterns = [
    "build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for autodoc -----------------------------------------------------

# Docstrings throughout the package are written in Google style, which
# sphinx.ext.napoleon translates for autodoc.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"

# Several constructors take a large number of annotated arguments. Rendering
# the annotations as part of the parameter descriptions keeps the signature
# lines readable.
autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # TabularDataset is used through the mapping protocol, so these are part
    # of its public interface.
    "special-members": "__getitem__, __getitems__, __len__",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# -- Options for nbsphinx ----------------------------------------------------

# The example notebooks read data and model snapshots that are not part of the
# repository, so they are rendered as-is instead of being executed. Notebooks
# committed with stored outputs display those outputs.
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
