import numpy as np

from voice_embedding import app
from voice_embedding.app import IFeatureEncoder, VoiceArray


def test__cast_bytes_to_voice_array(sample_bytes: bytes) -> None:
    features = app.cast_bytes_to_voice_array(data_bytes=sample_bytes)
    assert features.dtype == np.float32
    assert features.shape == (45920, 1)


def test__encode_voice_array(sample_voice_array: VoiceArray) -> None:
    features = app.encode_voice_array(voice_array=sample_voice_array)
    assert features.dtype == np.float32
    assert features.shape == (1, 287, 80)


def test__cast_embeddings_to_list() -> None:
    embedding = np.ones((1, 256), dtype=np.float32)
    embedding_list = app.cast_embeddings_to_list(embedding_array=embedding)
    assert embedding_list == [1.0] * 256


def test__encode(mock_onnx_provider: IFeatureEncoder, sample_bytes: bytes) -> None:
    embedding_list = app.encode(
        encoder_provider=mock_onnx_provider,
        data_bytes=sample_bytes,
    )
    assert embedding_list == [1.0] * 256
