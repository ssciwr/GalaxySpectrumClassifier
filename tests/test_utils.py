import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from GalaxySpectrumClassifier import TabularDataset
from GalaxySpectrumClassifier.utils import load_type, resolve_type_kwargs, to_xy


def test_load_type_module_level_name():
    assert load_type("sklearn.ensemble.RandomForestClassifier") is (
        RandomForestClassifier
    )
    assert load_type("numpy.abs") is np.abs


def test_load_type_resolves_attributes_nested_below_a_module():
    # The module/attribute boundary is not at the last dot here - the
    # importable prefix is only "pandas".
    assert load_type("pandas.DataFrame.dropna") is pd.DataFrame.dropna
    assert load_type("sklearn.ensemble.RandomForestClassifier.fit") is (
        RandomForestClassifier.fit
    )


def test_load_type_unknown_attribute_raises_attributeerror():
    with pytest.raises(AttributeError):
        load_type("pandas.does_not_exist")

    with pytest.raises(AttributeError):
        load_type("pandas.DataFrame.does_not_exist")


def test_load_type_unknown_module_raises_modulenotfound():
    with pytest.raises(ModuleNotFoundError):
        load_type("not_a_real_package.submodule.Thing")


def test_resolve_type_kwargs_leaves_plain_values_unchanged():
    kwargs = {"n_estimators": 10, "random_state": 42, "name": "rf"}

    assert resolve_type_kwargs(kwargs) == kwargs


def test_resolve_type_kwargs_resolves_type_spec():
    kwargs = {
        "estimator": {"type": "sklearn.ensemble.RandomForestClassifier"},
        "n_estimators": 10,
    }

    resolved = resolve_type_kwargs(kwargs)

    assert resolved["estimator"] is RandomForestClassifier
    assert resolved["n_estimators"] == 10


def test_resolve_type_kwargs_resolves_bare_type_string():
    resolved = resolve_type_kwargs(
        {
            "module": "torch.nn.Linear",
            "average": "binary",
        }
    )

    assert resolved["module"] is torch.nn.Linear
    assert resolved["average"] == "binary"


def test_resolve_type_kwargs_ignores_dicts_with_extra_keys():
    # Only a dict shaped *exactly* {"type": ...} is treated as a type
    # reference - anything else (e.g. a dict that happens to have a "type"
    # key among others) is passed through unchanged.
    kwargs = {
        "estimator": {"type": "sklearn.ensemble.RandomForestClassifier", "extra": 1}
    }

    assert resolve_type_kwargs(kwargs) == kwargs


def test_resolve_type_kwargs_empty():
    assert resolve_type_kwargs({}) == {}


def test_to_xy_class_map_missing_label_raises(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    pd.DataFrame({"a": [1.0, 2.0], "source": ["agn", "unknown"]}).to_csv(
        datapath / "0.dat", index=False
    )
    dataset = TabularDataset(datapath, read_kwargs={"sep": ","}, suffix=".dat")

    with pytest.raises(KeyError):
        to_xy(dataset, class_map={"agn": 0})


def test_to_xy_unknown_task_raises(create_data):
    dataset = TabularDataset(create_data, read_kwargs={"sep": ","}, suffix=".dat")

    with pytest.raises(ValueError, match="unknown task"):
        to_xy(dataset, task="clustering", label_column="d")
