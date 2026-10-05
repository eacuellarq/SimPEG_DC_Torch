"""
dctorch — DC/ERT resistivity on GPU with differentiable forward modeling.

Extends stock SimPEG (>=0.25) from the OUTSIDE: nothing in site-packages is
patched. Torch lives entirely inside this package; simpeg/discretize stay numpy.

Phase 1: sparse direct solver (cuDSS via nvmath) as a torch autograd op.
"""

__version__ = "0.2.0"

from .solver import (CuDSSBatchSolver, CuDSSSolver,  # noqa: F401
                     sparse_solve, sparse_solve_batch)
from .simulation2d import TorchDC2D  # noqa: F401
from .ip2d import TorchIP2D  # noqa: F401
from .ip3d import TorchIP3D  # noqa: F401
from .dcip2d import TorchDCIP2D  # noqa: F401
from .cr2d import TorchCR2D  # noqa: F401
from .simulation3d import TorchDC3D  # noqa: F401
from .optimization import TorchGaussNewton, TorchLBFGS  # noqa: F401
from .objectives import StudentTDataMisfit  # noqa: F401
