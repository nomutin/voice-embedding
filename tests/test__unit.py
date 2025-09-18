from pathlib import Path

import numpy as np

import voice_embedding
from voice_embedding import OnnxFeatureEncoderProvider, S3Provider


def test__cast_bytes_to_numpy_features() -> None:
    data_bytes = Path("tests/sample.wav").read_bytes()
    features = voice_embedding.cast_bytes_to_numpy_features(data_bytes)
    assert features.dtype == np.float32
    assert features.shape == (1, 287, 80)


def test__cast_numpy_embeddings_to_list() -> None:
    embedding = np.ones((1, 256), dtype=np.float32)
    embedding_list = voice_embedding.cast_numpy_embeddings_to_list(data=embedding)
    assert embedding_list == [1.0] * 256


def test__encode(
    mock_s3_provider: S3Provider,
    mock_onnx_provider: OnnxFeatureEncoderProvider,
) -> None:
    embedding_list = voice_embedding.encode(
        s3_provider=mock_s3_provider,
        encoder_provider=mock_onnx_provider,
        bucket="test-bucket",
        key="test-key",
    )
    assert embedding_list == [1.0] * 256
