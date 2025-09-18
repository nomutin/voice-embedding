import io
from pathlib import Path
from typing import Annotated

import boto3
import numpy as np
import onnxruntime
import soundfile
import soxr
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.parser import event_parser
from aws_lambda_powertools.utilities.streaming.s3_object import S3Object
from aws_lambda_powertools.utilities.typing import LambdaContext
from einops import pack, rearrange, reduce
from kaldi_native_fbank import FbankOptions, OnlineFbank
from numpy import typing as npt
from pydantic import BaseModel

logger = Logger()

type Feature = Annotated[npt.NDArray[np.float32], "1 length 80"]
type Embedding = Annotated[npt.NDArray[np.float32], "1 256"]


class RequestPayload(BaseModel):
    bucket: str
    key: str


class S3Provider:
    def __init__(self, session: boto3.Session) -> None:
        self.client = session.client("s3")

    def get_object(self, bucket: str, key: str) -> bytes:
        s3 = S3Object(bucket=bucket, key=key, boto3_client=self.client)
        return s3.read()


class OnnxFeatureEncoderProvider:
    def __init__(self, model_path: Path) -> None:
        self.session = onnxruntime.InferenceSession(model_path)

    def infer(self, voice_features: Feature) -> Embedding:
        outputs = self.session.run(None, {"feats": voice_features})
        if not isinstance((output := outputs[0]), np.ndarray):
            logger.error("Type Error", outputs=outputs)
            raise TypeError
        return output


def cast_bytes_to_numpy_features(data_bytes: bytes) -> Feature:
    with io.BytesIO(data_bytes) as f:
        data_numpy, fs = soundfile.read(f, dtype="float32", always_2d=True)
    if not isinstance(data_numpy, np.ndarray):
        logger.error("Type Error", data=data_numpy)
        raise TypeError
    data_numpy = soxr.resample(data_numpy, in_rate=fs, out_rate=16000)
    data_numpy = reduce(data_numpy, "length channels -> length", reduction="mean")
    data_numpy *= 1 << 15

    opts = FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False
    fbank = OnlineFbank(opts)

    fbank.accept_waveform(sampling_rate=16000, waveform=data_numpy.tolist())
    fbank.input_finished()
    features, _ = pack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], "* dim")
    features -= reduce(features, "length dim -> dim", reduction="mean")
    return rearrange(features, "length dim -> 1 length dim")


def cast_numpy_embeddings_to_list(data: Embedding) -> list[float]:
    data_flatten = rearrange(data, "1 dim -> dim")
    data_list = data_flatten.tolist()
    if not isinstance(data_list, list):
        logger.error("Type Error", data=data_list)
        raise TypeError
    return data_list


def encode(
    s3_provider: S3Provider,
    encoder_provider: OnnxFeatureEncoderProvider,
    bucket: str,
    key: str,
) -> list[float]:
    data_bytes = s3_provider.get_object(bucket=bucket, key=key)
    features = cast_bytes_to_numpy_features(data_bytes=data_bytes)
    output = encoder_provider.infer(voice_features=features)
    return cast_numpy_embeddings_to_list(data=output)


@event_parser(model=RequestPayload)  # type: ignore[misc]
def lambda_handler(event: RequestPayload, _context: LambdaContext) -> list[float]:
    try:
        s3_provider = S3Provider(session=boto3.Session())
        model_path = Path("voxceleb_resnet34_LM.onnx")
        onnx_provider = OnnxFeatureEncoderProvider(model_path=model_path)
        return encode(
            s3_provider=s3_provider,
            encoder_provider=onnx_provider,
            bucket=event.bucket,
            key=event.key,
        )
    except Exception as exc:
        logger.exception("Unhandled exception in lambda_handler", exc_info=exc)
        raise
