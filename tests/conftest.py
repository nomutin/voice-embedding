from pathlib import Path
from typing import override

import numpy as np
import pytest

from voice_embedding.app import FeatureArray, IFeatureEncoder, VoiceArray


class MockOnnxVoiceFeatureProvider(IFeatureEncoder):
    @override
    def __init__(self, model_path: Path) -> None:
        pass

    @override
    def encode(self, feature: VoiceArray) -> FeatureArray:
        return np.ones((1, 256), dtype=np.float32)


@pytest.fixture
def mock_onnx_provider() -> IFeatureEncoder:
    return MockOnnxVoiceFeatureProvider(model_path=Path())


@pytest.fixture
def sample_bytes() -> bytes:
    return Path("tests/sample.wav").read_bytes()


@pytest.fixture
def sample_voice_array() -> VoiceArray:
    return np.zeros([45920, 2]).astype(np.float32)


@pytest.fixture
def sample_feature_array() -> FeatureArray:
    return np.zeros([1, 287, 80]).astype(np.float32)
