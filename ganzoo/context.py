"""The CONTEXT module contains inner aux functions."""
import os
import sys
import inspect
from absl import app

def main(fn):
    """Call fn with command line arguments.  Used as a decorator.

    The main decorator marks the function that starts a program. For example,

    @main
    def my_run_function():
        # function body

    Use this instead of the typical __name__ == "__main__" predicate.
    """
    if inspect.stack()[1][0].f_locals['__name__'] == '__main__':
        # Pass flags by absl.FLAGS
        app.run(fn)
    return fn

_ROOTDIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '../',
    )
)
_DATADIR = os.path.join(_ROOTDIR, 'data')
_WORKDIR = os.path.join(_ROOTDIR, 'exps')
