import contextlib
import os
import sys
import warnings

import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from IPython.core.magic import register_cell_magic
from IPython.display import HTML, display
from matplotlib import pyplot as plt

# https://github.com/ipython/ipython/issues/13262
sys.breakpointhook = set_trace
in_vscode = "VSCODE_CWD" in os.environ
snippets = {}


try:

    @register_cell_magic
    def snippet(line, cell):
        snippets[line.strip()] = cell

except NameError:
    pass


def df(index=None, **kwargs):
    return pd.DataFrame(kwargs, index)


def df_size(df, deep=True, unit="mb", decimals=2):
    exp = {"kb": 1, "mb": 2, "gb": 3}[unit]
    return round(df.memory_usage(deep=True).sum() / 1024**exp, decimals)


def display_all(value, max_rows=None, max_columns=None):
    with pd.option_context(
        "display.max_rows", max_rows, "display.max_columns", max_columns
    ):
        display(value)


def config_pandas(v3_compat=True):
    if v3_compat:
        # https://pandas.pydata.org/docs/user_guide/migration-3-strings.html
        pd.options.future.infer_string = True
        # https://pandas.pydata.org/docs/user_guide/copy_on_write.html
        pd.options.mode.copy_on_write = True
        pd.options.future.no_silent_downcasting = True


def keep_quiet(*filters):
    for kwargs in [
        dict(message=r".*Consider using BigQuery DataFrames.*LargeResultsWarning"),
        *filters,
    ]:
        warnings.filterwarnings("ignore", **kwargs)

    # https://stackoverflow.com/a/78793583
    np.set_printoptions(legacy="1.25")
    # https://share.google/aimode/EhpP13AsS1FJKPSD0
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


def nice_style(md_width="80ch", figsize=(6, 4)):
    # https://github.com/microsoft/vscode-jupyter/issues/11175
    # https://stackoverflow.com/questions/58801176
    # https://stackoverflow.com/questions/57442034
    # https://stackoverflow.com/questions/77555651
    if in_vscode:
        css = """
        <style>
          .dataframe th {font-family: "Source Code Pro";}
          .dataframe tr {font-family: "Source Code Pro";}
          .markup .preview {width: %s !important; }
        </style>
        """
        display(HTML(css % (md_width,)))
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.figsize"] = figsize


@contextlib.contextmanager
def ignore_warnings(category):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=category)
        yield


def plot_diagonal(ax=None):
    (ax or plt).axline((0, 0), slope=1, color="k", linestyle=":", alpha=0.3)
