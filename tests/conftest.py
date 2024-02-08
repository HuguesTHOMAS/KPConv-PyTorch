from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixture_path():
    yield Path(__file__).parent / "fixtures"


@pytest.fixture
def ply_array():
    yield np.array([[0, 10, 200], [2, 30, 400], [7, 10, 600], [5, 60, 200]], dtype=np.float64)
