import os
import sys
# Configuration file for the Sphinx documentation builder.
sys.path.insert(0, os.path.abspath('../')) 

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Project2'
copyright = '2025, alagis'
author = 'alagis'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Автоматически читает docstrings
    'sphinx.ext.viewcode',      # Показывает исходный код
    'sphinx.ext.napoleon',      # Поддержка Google-style docstrings
    'sphinx_autodoc_typehints', # Для аннотаций типов
    'recommonmark',             # Поддержка Markdown
    'sphinx.ext.autosummary'
]

autosummary_generate = True     # Генерировать .rst файлы автоматически
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': False,
    'inherited-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_baseurl = "https://analagis.github.io/learning_ML_projects/" 