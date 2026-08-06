import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def build_data():
    """Build ten 100-row frames of synthetic samples.

    Returns:
        list[pd.DataFrame]: One frame per file the fixtures will write, each
            with four float feature columns and two integer label columns.
    """
    rng = np.random.default_rng(42)
    data = []
    for _ in range(0, 10):
        a = rng.uniform(0, 100, size=100)
        b = rng.normal(0, 1, size=100)
        c = rng.uniform(-100, 100, size=100)
        d = rng.uniform(5, 10, size=100)
        # Integer-valued target, so tests exercise the label split without
        # relying on the float -> int64 cast.
        source = rng.integers(0, 2, size=100)
        extra = rng.integers(0, 3, size=100)

        df = pd.DataFrame(
            {"a": a, "b": b, "c": c, "d": d, "source": source, "extra": extra}
        )
        data.append(df)
    return data


@pytest.fixture
def create_data(tmp_path):
    """Write the synthetic frames as ten comma-separated ``.dat`` files.

    Real files on disk rather than an in-memory stand-in, so tests exercise
    ``TabularDataset``'s own reading and per-file indexing.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path: The directory to hand to ``TabularDataset(path=...)``. Note the
            files carry the frame index as an unnamed first column, so a
            dataset reading them without ``read_kwargs={"index_col": 0}`` sees
            one extra feature.
    """
    datapath = tmp_path / "data"
    datapath.mkdir()
    data = build_data()
    for i, df in enumerate(data):
        df.to_csv(Path(datapath) / f"{i}.dat")

    return datapath


@pytest.fixture
def create_data_nonstandard(tmp_path):
    """Write the same frames as tab-separated ``.tsv`` files wrapped in comments.

    Covers the non-default reader settings: a ``\\t`` separator, ``//`` comment
    markers before and after the data, and a suffix other than ``.dat``.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path: The directory to hand to ``TabularDataset(path=...)``.
    """
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
