from collections.abc import Generator
from pathlib import Path
from typing import override

import boto3
import numpy as np
import pytest
from moto import mock_aws

from voice_embedding import Embedding, Feature, OnnxFeatureEncoderProvider, S3Provider


@mock_aws
class MockS3Provider(S3Provider):
    @override
    def __init__(self, session: boto3.Session) -> None:
        super().__init__(session)
        self.client.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-northeast-1"},
        )
        sample_data = Path("tests/sample.wav").read_bytes()
        self.client.put_object(
            Bucket="test-bucket",
            Key="test-key",
            Body=sample_data,
        )


class MockOnnxVoiceFeatureProvider(OnnxFeatureEncoderProvider):
    @override
    def __init__(self, model_path: Path) -> None:
        pass

    @override
    def infer(self, voice_features: Feature) -> Embedding:
        return np.ones((1, 256), dtype=np.float32)


@pytest.fixture
def mock_onnx_provider() -> MockOnnxVoiceFeatureProvider:
    return MockOnnxVoiceFeatureProvider(model_path=Path())


@pytest.fixture
def mock_s3_provider() -> Generator[S3Provider]:
    with mock_aws():
        session = boto3.Session(region_name="ap-northeast-1")
        yield MockS3Provider(session=session)
