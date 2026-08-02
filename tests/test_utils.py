from sklearn.ensemble import RandomForestClassifier

from GalaxySpectrumClassifier.utils import resolve_type_kwargs


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
