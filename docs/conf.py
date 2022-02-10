#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("numba_2pcf").version
except DistributionNotFound:
    __version__ = "dev"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
]
autodoc_mock_imports = []

project = "numba-2pcf"
copyright = "2022 Lehman Garrison"
version = __version__
release = __version__

exclude_patterns = ["_build"]
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_title = "numba-2pcf"
html_show_sourcelink = False
html_baseurl = "https://numba_2pcf.readthedocs.io/en/latest/"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/lgarrison/numba-2pcf",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
