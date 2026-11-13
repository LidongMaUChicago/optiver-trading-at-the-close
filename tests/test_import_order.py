import subprocess
import sys


def test_torch_loads_before_xgboost():
    """Regression test for the real-data LSTM SIGSEGV (competing libomp runtimes on macOS)."""
    code = ("import sys, optiver_close, optiver_close.models; "
            "names = list(sys.modules); "
            "print(names.index('torch') < names.index('xgboost'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"
