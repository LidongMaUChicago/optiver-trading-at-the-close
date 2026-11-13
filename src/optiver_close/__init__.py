"""Optiver closing-auction research, rebuilt from Lidong Ma's 2023 notebooks."""
import os

# Bound native thread pools before numpy/sklearn/xgboost/torch are imported.
for _name, _value in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "2",
                      "MKL_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "1",
                      "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
    os.environ.setdefault(_name, _value)

# On macOS wheels, scikit-learn, xgboost (Homebrew) and torch each ship their own
# libomp. If xgboost's runtime is initialised before torch's, torch's CPU kernels
# segfault on real data (observed in AdamW.step, exit -11). Loading torch first
# makes the import order deterministic; torch stays an optional dependency.
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _torch = None

__version__ = "0.1.0"
