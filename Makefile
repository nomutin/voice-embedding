.PHONY: init lint format test deploy

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

deploy:
	mkdir -p model
	curl -L https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx -o model/voxceleb_resnet34_LM.onnx
	zip -r model.zip model
	aws s3 cp model.zip s3://voice-embedding/model.zip
	rm -r model
	rm model.zip
	@uv export --no-editable --no-hashes --no-dev --output-file src/voice_embedding/requirements.txt --no-emit-project --no-annotate --no-header --quiet
	@rm -rf .aws-sam
	uv run sam build --use-container --template aws/template.yaml
	uv run sam deploy --config-file aws/samconfig.toml
