import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def build_data():
    rng = np.random.default_rng(42)
    data = []
    for _ in range(0, 10):
        a = rng.uniform(0, 100, size=100)
        b = rng.normal(0, 1, size=100)
        c = rng.uniform(-100, 100, size=100)
        d = rng.uniform(5, 10, size=100)

        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
        data.append(df)
    return data


@pytest.fixture
def create_data(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    data = build_data()
    for i, df in enumerate(data):
        df.to_csv(Path(datapath) / f"{i}.dat")

    return datapath


@pytest.fixture
def create_data_nonstandard(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    data = build_data()
    for i, df in enumerate(data):
        with open(Path(datapath) / f"{i}.tsv", mode="x") as file:
            file.write("// comment below\n")
            file.write("\n")

        df.to_csv(Path(datapath) / f"{i}.tsv", sep="\t", mode="a")

        with open(Path(datapath) / f"{i}.tsv", mode="a") as file:
            file.write("// comment below")

    return datapath
