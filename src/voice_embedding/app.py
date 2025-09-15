import io
import json
from pathlib import Path
from typing import Annotated, Literal

import boto3
import numpy as np
import onnxruntime
import soundfile
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.parser import event_parser
from aws_lambda_powertools.utilities.streaming.s3_object import S3Object
from aws_lambda_powertools.utilities.typing import LambdaContext
from einops import rearrange, reduce
from kaldi_native_fbank import FbankOptions, OnlineFbank
from numpy import typing as npt
from pydantic import BaseModel

logger = Logger()


class RequestPayload(BaseModel):
    bucket: str
    key: str


class ResponsePayload(BaseModel):
    status_code: Literal[200, 400, 500]
    body: str


class S3Provider:
    def __init__(self, session: boto3.Session) -> None:
        self.client = session.client("s3")

    def get_object(self, bucket: str, key: str) -> bytes:
        s3 = S3Object(bucket=bucket, key=key, boto3_client=self.client)
        return s3.read()


class OnnxVoiceFeatureEncoder:
    def __init__(self, model_path: Path) -> None:
        self.session = onnxruntime.InferenceSession(model_path)

    def infer(
        self,
        voice_features: Annotated[npt.NDArray[np.float32], "1 length 80"],
    ) -> Annotated[npt.NDArray[np.float32], "1 1 192"]:
        outputs = self.session.run(None, {"input": voice_features})
        if not isinstance((output := outputs[0]), np.ndarray):
            raise TypeError
        return output


def cast_bytes_to_numpy_features(data_bytes: bytes) -> Annotated[npt.NDArray[np.float32], "1 length 80"]:
    with io.BytesIO(data_bytes) as f:
        data_numpy, _fs = soundfile.read(f, dtype="float32", always_2d=True)
    if not isinstance(data_numpy, np.ndarray):
        raise TypeError
    data_numpy = reduce(data_numpy, "length channels -> length", reduction="mean")

    opts = FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False
    fbank = OnlineFbank(opts)
    fbank.accept_waveform(sampling_rate=16000, waveform=data_numpy.tolist())
    fbank.input_finished()
    features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], axis=0)
    return rearrange(features, "length dim -> 1 length dim")


def cast_numpy_embeddings_to_list(data: Annotated[npt.NDArray[np.float32], "1 1 192"]) -> list[float]:
    data_flatten = rearrange(data, "1 1 dim -> dim")
    data_list = data_flatten.tolist()
    if not isinstance(data_list, list):
        raise TypeError
    return data_list


def encode(
    s3_provider: S3Provider,
    encoder_provider: OnnxVoiceFeatureEncoder,
    bucket: str,
    key: str,
) -> list[float]:
    data_bytes = s3_provider.get_object(bucket=bucket, key=key)
    features = cast_bytes_to_numpy_features(data_bytes=data_bytes)
    output = encoder_provider.infer(voice_features=features)
    return cast_numpy_embeddings_to_list(data=output)


@logger.inject_lambda_context(log_event=True)
@event_parser(model=RequestPayload)
def lambda_handler(event: RequestPayload, _context: LambdaContext) -> ResponsePayload:
    try:
        s3_provider = S3Provider(session=boto3.Session())
        onnx_provider = OnnxVoiceFeatureEncoder(model_path=Path("/opt/ml/model/voice_feature_encoder.onnx"))
        embeddings = encode(
            s3_provider=s3_provider,
            encoder_provider=onnx_provider,
            bucket=event.bucket,
            key=event.key,
        )
    except TypeError:
        logger.exception("TypeError")
        return ResponsePayload(
            status_code=400,
            body=json.dumps({"error": "Bad request"}),
        )
    except Exception:
        logger.exception("Exception")
        return ResponsePayload(
            status_code=500,
            body=json.dumps({"error": "Internal server error"}),
        )
    else:
        return ResponsePayload(
            status_code=200,
            body=json.dumps({"embeddings": embeddings}),
        )
