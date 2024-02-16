from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixture_path():
    yield Path(__file__).parent / "fixtures"


@pytest.fixture
def points_array():
    yield np.array(
        [
            [5.83, -18.83, 0.24],
            [5.83, -18.86, 0.22],
            [5.82, -18.84, 0.24],
            [5.83, -18.84, 0.23],
            [5.82, -18.88, 0.23],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def colors_array():
    yield np.array(
        [
            [158, 167, 176],
            [151, 162, 176],
            [157, 164, 176],
            [155, 165, 176],
            [154, 167, 177],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def classification_array():
    yield np.array(
        [1, 2, 2, 3, 1],
        dtype=np.int32,
    )


@pytest.fixture
def dataset():
    yield "S3DIS"
