import os
import shutil

import pytest

from ECOv003_L2T_STARS.julia_environment import (
    JuliaEnvironmentError,
    PROJECT_SCRIPTS,
    verify_julia_environments,
)


def test_julia_scripts_and_manifests_exist():
    for project, scripts in PROJECT_SCRIPTS.items():
        assert os.path.exists(os.path.join(project, "Project.toml"))
        assert os.path.exists(os.path.join(project, "Manifest.toml"))
        for script in scripts:
            assert os.path.exists(script)


# Opt-in: only meaningful where the pinned Julia packages are installed (e.g. the Docker image).
# Run with: CHECK_JULIA_ENV=1 pytest tests/test_julia_environment.py
@pytest.mark.skipif(
    os.environ.get("CHECK_JULIA_ENV") != "1" or shutil.which("julia") is None,
    reason="set CHECK_JULIA_ENV=1 and install julia to check the Julia environment",
)
def test_julia_imports():
    try:
        verify_julia_environments()
    except JuliaEnvironmentError as e:
        pytest.fail(str(e))
