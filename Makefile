.PHONY: init lint format test test-integration

init:
	- uv sync
	- curl -L https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx -o src/voice_embedding/voxceleb_resnet34_LM.onnx

lint:
	- uv run ruff check .
	- uv run mypy --strict .

format:
	- uv run ruff check . --fix
	- uv run ruff format .

test:
	- uv run pytest --cov=src --cov-report=term-missing

test-integration:
	- uv run pytest tests/integration
