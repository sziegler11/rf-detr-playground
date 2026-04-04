.PHONY: install download-data train evaluate predict benchmark visualize test clean

CONFIG ?= configs/aquarium.yaml
MODEL ?= base
CHECKPOINT ?= outputs/latest/best.pt
DATASET ?= aquarium
IMAGES ?= data/$(DATASET)/valid

install:
	pip install -e ".[all]"

download-data:
	python scripts/download_datasets.py --all

download-dataset:
	python scripts/download_datasets.py --dataset $(DATASET)

train:
	python scripts/train.py --config $(CONFIG) --model $(MODEL)

evaluate:
	python scripts/evaluate.py --checkpoint $(CHECKPOINT) --dataset $(DATASET)

predict:
	python scripts/predict.py --checkpoint $(CHECKPOINT) --images $(IMAGES)

benchmark:
	python scripts/benchmark.py

visualize:
	python scripts/visualize.py --results-dir outputs/

test:
	pytest tests/ -v

clean:
	rm -rf outputs/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
