from pathlib import Path

import numpy as np

from voice_embedding import app


def test__cast_bytes_to_numpy_features() -> None:
    data_bytes = Path("tests/sample.wav").read_bytes()
    features = app.cast_bytes_to_numpy_features(data_bytes)
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (1, 287, 80)


def test__cast_numpy_embeddings_to_list() -> None:
    embedding = np.ones((1, 1, 192), dtype=np.float32)
    embedding_list = app.cast_numpy_embeddings_to_list(data=embedding)
    assert isinstance(embedding_list, list)
    assert len(embedding_list) == 192
    assert all(x == 1.0 for x in embedding_list)


def test__encode(
    mock_s3_provider: app.S3Provider,
    mock_onnx_provider: app.OnnxVoiceFeatureEncoder,
) -> None:
    embedding_list = app.encode(
        s3_provider=mock_s3_provider,
        encoder_provider=mock_onnx_provider,
        bucket="test-bucket",
        key="test-key",
    )
    assert isinstance(embedding_list, list)
    assert len(embedding_list) == 192
    assert all(x == 1.0 for x in embedding_list)
