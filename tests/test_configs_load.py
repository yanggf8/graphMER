from pathlib import Path
import yaml


def test_load_train_cpu_yaml():
    path = Path("configs/train_cpu.yaml")
    data = yaml.safe_load(path.read_text())
    assert data["hardware"]["device"] == "cpu"
    assert data["training_data"]["max_seq_len"] >= 384


def test_load_train_tpu_yaml():
    path = Path("configs/train_tpu.yaml")
    data = yaml.safe_load(path.read_text())
    assert data["hardware"]["device"] == "tpu"
    assert data["run"]["mixed_precision"] == "bf16"
