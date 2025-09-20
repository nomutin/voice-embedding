import base64
import io
import json
from pathlib import Path
from typing import Annotated, Protocol, TypedDict

import numpy as np
import onnxruntime
import soundfile
import soxr
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent, event_source
from aws_lambda_powertools.utilities.typing import LambdaContext
from einops import pack, rearrange, reduce
from kaldi_native_fbank import FbankOptions, OnlineFbank
from numpy import typing as npt

logger = Logger()

type VoiceArray = Annotated[npt.NDArray[np.float32], "length channels"]
type FeatureArray = Annotated[npt.NDArray[np.float32], "1 length 80"]
type EmbeddingArray = Annotated[npt.NDArray[np.float32], "1 256"]


class ResponsePayload(TypedDict):
    statusCode: int
    body: str


class IFeatureEncoder(Protocol):
    def encode(self, feature: FeatureArray) -> EmbeddingArray: ...


class OnnxFeatureEncoderProvider(IFeatureEncoder):
    def __init__(self, model_path: Path) -> None:
        self.session = onnxruntime.InferenceSession(model_path)

    def encode(self, feature: FeatureArray) -> EmbeddingArray:
        outputs = self.session.run(None, {"feats": feature})
        if not isinstance((output := outputs[0]), np.ndarray):
            logger.error("Type Error", outputs=outputs)
            raise TypeError
        return output


def cast_bytes_to_voice_array(data_bytes: bytes) -> VoiceArray:
    with io.BytesIO(data_bytes) as f:
        data_numpy, fs = soundfile.read(f, dtype="float32", always_2d=True)
        data_numpy = soxr.resample(data_numpy, in_rate=fs, out_rate=16000)
    if not isinstance(data_numpy, np.ndarray):
        logger.error("Type Error", data=data_numpy)
        raise TypeError
    return data_numpy


def encode_voice_array(voice_array: VoiceArray) -> FeatureArray:
    opts = FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False
    fbank = OnlineFbank(opts)

    voice_array = reduce(voice_array, "length channels -> length", reduction="mean")
    fbank.accept_waveform(sampling_rate=16000, waveform=voice_array.tolist())
    fbank.input_finished()
    features, _ = pack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], "* dim")
    features -= reduce(features, "length dim -> dim", reduction="mean")
    return rearrange(features, "length dim -> 1 length dim")


def cast_embeddings_to_list(embedding_array: EmbeddingArray) -> list[float]:
    embedding_list = rearrange(embedding_array, "1 dim -> dim").tolist()
    if not isinstance(embedding_list, list):
        logger.error("Type Error", embedding_list=embedding_list)
        raise TypeError
    return embedding_list


def encode(encoder_provider: IFeatureEncoder, data_bytes: bytes) -> list[float]:
    voice_array = cast_bytes_to_voice_array(data_bytes=data_bytes)
    feature_array = encode_voice_array(voice_array=voice_array)
    embedding_array = encoder_provider.encode(feature=feature_array)
    return cast_embeddings_to_list(embedding_array=embedding_array)


@event_source(data_class=LambdaFunctionUrlEvent)  # type: ignore[misc]
def lambda_handler(event: LambdaFunctionUrlEvent, _context: LambdaContext) -> ResponsePayload:
    try:
        if event.http_method != "POST" or event.body is None:
            return ResponsePayload(statusCode=400, body="Bad Request")

        model_path = Path("/opt/model/voxceleb_resnet34_LM.onnx")
        onnx_provider = OnnxFeatureEncoderProvider(model_path=model_path)
        embedding = encode(
            encoder_provider=onnx_provider,
            data_bytes=base64.b64decode(event.body),
        )
        return ResponsePayload(statusCode=200, body=json.dumps(embedding))
    except Exception as exc:
        logger.exception("Unhandled exception in lambda_handler", exc_info=exc)
        raise
