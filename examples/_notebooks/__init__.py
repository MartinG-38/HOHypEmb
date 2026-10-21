"""Cells of the example notebooks, one module per notebook, assembled by ``examples/build_examples.py``."""
from textwrap import dedent

from nbformat.v4 import new_code_cell, new_markdown_cell


def md(text: str):
    """Markdown cell from an indented triple-quoted string (dedented, stripped)."""
    return new_markdown_cell(dedent(text).strip())


def code(text: str):
    """Code cell from an indented triple-quoted string (dedented, stripped)."""
    return new_code_cell(dedent(text).strip())
