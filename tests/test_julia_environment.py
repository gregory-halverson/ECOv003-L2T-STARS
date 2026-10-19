import shutil
import subprocess
from pathlib import Path

import pytest

SMOKE_TEST = Path(__file__).parent.parent / "ECOv003_L2T_STARS" / "julia_smoke_test.jl"


@pytest.mark.skipif(shutil.which("julia") is None, reason="julia not installed")
def test_julia_imports():
    result = subprocess.run(
        ["julia", str(SMOKE_TEST)], capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stdout + result.stderr
