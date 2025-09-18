from pathlib import Path

import pytest

import voice_embedding
from voice_embedding import OnnxFeatureEncoderProvider, S3Provider


@pytest.mark.integration
def test__encode_with_model(mock_s3_provider: S3Provider) -> None:
    model_path = Path("src/voice_embedding/voxceleb_resnet34_LM.onnx")
    encoder_provider = OnnxFeatureEncoderProvider(model_path=model_path)
    embedding_list = voice_embedding.encode(
        s3_provider=mock_s3_provider,
        encoder_provider=encoder_provider,
        bucket="test-bucket",
        key="test-key",
    )
    assert isinstance(embedding_list, list)
