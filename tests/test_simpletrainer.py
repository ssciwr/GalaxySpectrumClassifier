import pytest
import torch
import skorch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from GalaxySpectrumClassifier import SimpleTrainer


@pytest.fixture
def random_forest_model():
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def skorch_torch_model():
    class SimpleNN(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    input_dim = 20
    output_dim = 2
    model = SimpleNN(input_dim, output_dim)
    return skorch.NeuralNetClassifier(model, max_epochs=10, lr=0.1)


@pytest.fixture
def synthetic_dataset():
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


def test_simple_trainer_init_minimal():
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )

    assert isinstance(trainer.model, RandomForestClassifier)
    assert trainer.model.n_estimators == 10
    assert trainer.model.random_state == 42
    assert trainer.metrics[0]["name"] == "accuracy_score"
    assert trainer.metrics[0]["callable"] == accuracy_score
    assert trainer.metrics[0]["args"] == []
    assert trainer.metrics[0]["kwargs"] == {}
    assert trainer.metrics[0]["needs_proba"] is False


def test_simple_trainr_from_config(): ...


def test_simple_trainer_fit(): ...


def test_simple_trainer_fit_with_calibrator(): ...


def test_simple_trainer_fit_torchmodel(): ...


def test_simple_trainer_fit_torchmodel_with_calibrator(): ...


def test_simple_trainer_predict(): ...


def test_simple_trainer_evaluate(): ...


def test_simple_trainer_save_load(): ...


def test_simple_trainer_with_custom_metric(): ...
