import tempfile
import shutil
import urllib.request
from contextlib import contextmanager
from pathlib import Path

_SNAP_BASE = "https://github.com/openbraininstitute/snap/raw/v3.0.3/tests/data"
_SNAP_FILES = ["circuit_config.json", "nodes.h5", "edges.h5", "node_sets.json"]


@contextmanager
def get_snap_test_circuit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for fname in _SNAP_FILES:
            urllib.request.urlretrieve(f"{_SNAP_BASE}/{fname}", tmp_dir / fname)
        yield tmp_dir / "circuit_config.json"
    finally:
        shutil.rmtree(tmp_dir)
