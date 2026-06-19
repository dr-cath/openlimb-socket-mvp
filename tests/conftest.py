"""Shared pytest fixtures.

Tests use the synthetic fallback limb by default so the suite is hermetic (no
network, no OpenLimbTT download required). OpenLimbTT-specific tests skip
themselves when the data is not present.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from socketcad.limb_source import generate_synthetic_limb, load_openlimb_model  # noqa: E402
from socketcad.geometry import estimate_frame  # noqa: E402


@pytest.fixture(scope="session")
def limb():
    """A reproducible synthetic residual-limb mesh."""
    return generate_synthetic_limb(seed=7)


@pytest.fixture(scope="session")
def frame(limb):
    return estimate_frame(limb)


@pytest.fixture(scope="session")
def openlimb():
    model = load_openlimb_model()
    if model is None:
        pytest.skip("OpenLimbTT data not present (run scripts/fetch_openlimb.py)")
    return model
