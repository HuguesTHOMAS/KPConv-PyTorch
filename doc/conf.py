#!python3

"""
    Configuration for Sphinx.
"""

# standard
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(r".."))

# -- Project information -----------------------------------------------------
author = "Hugues Thomas", "Oslandia"
description = "KPConv-Torch documentation"
project = "Preprocessing, training and inference of the model"
version = release = datetime.today().strftime("%Y.%m.%d")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Sphinx included
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    # 3rd party
    "myst_parser",
    "sphinx_copybutton",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
autosectionlabel_prefix_document = True
# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "fr"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    ".DS_Store",
    ".venv",
    "_archive",
    "_build",
    "_output",
    "ext_libs",
    "qgis_plugin_workshop",
    "README.md",
    "Thumbs.db",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# -- Theme

html_favicon = "_static/img/oslandia_logo.svg"
html_logo = "_static/img/oslandia_logo.svg"
html_theme = "furo"
# html_theme_options = {
#     "source_edit_link": f"{__about__.__uri_repository__}"
#     + "/-/edit/main/docs/{filename}",
# }

# -- EXTENSIONS --------------------------------------------------------

# Configuration for intersphinx (refer to others docs).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# MyST Parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
]

myst_heading_anchors = 3

myst_substitutions = {
    "author": author,
    "date_update": datetime.now().strftime("%d %B %Y"),
    "description": description,
    "title": project,
    "version": version,
}

myst_url_schemes = ["http", "https", "mailto"]
