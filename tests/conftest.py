import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def create_data(tmp_path):
    datapath = tmp_path / "data"
    rng = np.random.default_rng(42)

    for i in range(0, 10):
        a = rng.uniform(0, 100, size=100)
        b = rng.normal(0, 1, size=100)
        c = rng.uniform(-100, 100, size=100)
        d = rng.uniform(5, 10, size=100)

        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

        df.to_csv(Path(datapath) / f"{i}.csv")

    return datapath
