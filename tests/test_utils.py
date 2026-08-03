import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from GalaxySpectrumClassifier.utils import load_type, resolve_type_kwargs


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
