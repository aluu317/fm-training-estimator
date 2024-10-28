# Standard
from pathlib import Path
import pytest

# Third Party
from pytest import raises

# Local
from fm_training_estimator.config import parse
from fm_training_estimator.regressor import XGBoostRegressor
from fm_training_estimator.throughput.hybrid.hybrid import HybridSpeedEstimator

test_data2 = (Path(__file__).parent / "../../regressor/data_samples/data2.csv").as_posix()
test_data3 = (Path(__file__).parent / "../../regressor/data_samples/data3.csv").as_posix()


def test_hybrid_empty():
    fm, ta, ia, _, _ = parse({})

    with raises(RuntimeError):
        _ = HybridSpeedEstimator(fm, ta, ia, None, None)


def test_hybrid_lookup():
    fm, ta, ia, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "per_device_train_batch_size": 4,
            "block_size": 512,
            "numGpusPerPod": 2,
        }
    )

    est = HybridSpeedEstimator(fm, ta, ia, test_data2, None)

    assert est.get_tps() == 500
    # test lookup approach
    assert est.get_tps(1024) == 1000

@pytest.mark.skip(reason="Seg fault error")
def test_hybrid_reg(tmp_path):
    model_path = tmp_path / "test.model.json"
    reg = XGBoostRegressor()
    reg.train(test_data2, model_path, ["tokens_per_second", "memory", "memory_act"])

    fm, ta, ia, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "per_device_train_batch_size": 4,
            "block_size": 512,
            "numGpusPerPod": 4,
        }
    )

    est = HybridSpeedEstimator(fm, ta, ia, test_data2, model_path)

    assert est.get_tps() > 300

@pytest.mark.skip(reason="Seg fault error")
def test_hybrid_model_features(tmp_path):
    model_path = tmp_path / "test.model.json"
    reg = XGBoostRegressor()
    reg.train(test_data3, model_path, ["tokens_per_second", "memory", "memory_act"])

    fm, ta, ia, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "per_device_train_batch_size": 16,
            "block_size": 1024,
            "numGpusPerPod": 4,
        }
    )

    est = HybridSpeedEstimator(fm, ta, ia, test_data3, model_path)

    assert est.get_tps() > 400
