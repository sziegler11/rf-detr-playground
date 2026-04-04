"""Tests for the training module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from rf_detr_playground.training.trainer import TrainingConfig


class TestTrainingConfig:
    def test_default_config(self):
        cfg = TrainingConfig()
        assert cfg.model_size == "base"
        assert cfg.epochs == 50
        assert cfg.batch_size == 4
        assert cfg.grad_accum_steps == 4
        assert cfg.lr == 1e-4
        assert cfg.early_stopping is True

    def test_from_yaml(self, tmp_path):
        config_data = {
            "dataset_name": "aquarium",
            "model_size": "large",
            "epochs": 100,
            "batch_size": 8,
            "lr": 2e-4,
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = TrainingConfig.from_yaml(config_path)
        assert cfg.dataset_name == "aquarium"
        assert cfg.model_size == "large"
        assert cfg.epochs == 100
        assert cfg.batch_size == 8
        assert cfg.lr == 2e-4
        # Defaults should still be applied
        assert cfg.early_stopping is True

    def test_from_yaml_with_extra_keys(self, tmp_path):
        config_data = {
            "dataset_name": "coco128",
            "custom_key": "some_value",
            "another_param": 42,
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = TrainingConfig.from_yaml(config_path)
        assert cfg.dataset_name == "coco128"
        assert cfg.extra == {"custom_key": "some_value", "another_param": 42}

    def test_from_yaml_empty_file(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        cfg = TrainingConfig.from_yaml(config_path)
        assert cfg.model_size == "base"  # defaults apply

    def test_effective_batch_size(self):
        cfg = TrainingConfig(batch_size=4, grad_accum_steps=4)
        effective = cfg.batch_size * cfg.grad_accum_steps
        assert effective == 16
