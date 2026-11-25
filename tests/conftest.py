import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def yahoo_long(root):
    from btcregime.data import load_close
    return load_close(os.path.join(root, "data", "btc_yahoo_1d_long.csv"), "2014-09-17", "2026-10-31")
