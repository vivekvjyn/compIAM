import os
import sys
import numpy as np
from typing import Dict
from tqdm import tqdm
from compiam.exceptions import ModelNotTrainedError

from compiam.utils.download import download_remote_model
from compiam.utils import get_logger, WORKDIR
from compiam.io import write_csv
from compiam.rhythm.meter.beat_this_carnatic.post import Postprocessor
from compiam.rhythm.meter.beat_this_carnatic.pre import LogMelSpect, load_audio
from compiam.rhythm.meter.beat_this_carnatic.model import BeatThis
from compiam.rhythm.meter.beat_this_carnatic.utils import split_predict_aggregate
import torch
import torch.nn.functional as F
import soxr

logger = get_logger(__name__)


class BeatThisCarnatic(Audio2Frames):
    """
    Class for extracting beat and downbeat positions (in seconds) from an audio tensor.

    Args:
        checkpoint_path (str): Path to the model checkpoint file. It can be a local path, a URL, or a key from the CHECKPOINT_URL dictionary. Default is "final0", which will load the model trained on all data except GTZAN with seed 0.
        device (str): Device to use for inference. Default is "cpu".
        float16 (bool): Whether to use half precision floating point arithmetic. Default is False.
        dbn (bool): Whether to use the madmom DBN for post-processing. Default is False.
    """

    def __init__(
        self, checkpoint_path="final0", device="cpu", float16=False, dbn=False
    ):
        self.device = torch.device(device)
        self.float16 = float16
        self.model = BeatThis()
        self.frames2beats = Postprocessor(type="dbn" if dbn else "minimal")
        self.spect = LogMelSpect(device=self.device)

    def spect2frames(self, spect):
        with torch.inference_mode():
            with torch.autocast(enabled=self.float16, device_type=self.device.type):
                model_prediction = split_predict_aggregate(
                    spect=spect,
                    chunk_size=1500,
                    overlap_mode="keep_first",
                    border_size=6,
                    model=self.model,
                )
        return model_prediction["beat"].float(), model_prediction["downbeat"].float()

    def signal2spect(self, signal, sr):
        if signal.ndim == 2:
            signal = signal.mean(1)
        elif signal.ndim != 1:
            raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        return self.spect(signal)

    def _build_model(self):
        """Build the TCN model."""
        model = BeatThis().to(self.device)
        model.eval()
        return model

    def load_model(self, model_path):
        """Load pre-trained model weights."""
        if not os.path.exists(os.path.join(model_path, self.model_version)):
            self.download_model(model_path)  # Downloading model weights

        self.model.load_weights(os.path.join(model_path, self.model_version), self.device)

        self.model_path = model_path
        self.trained = True

    def download_model(self, model_path=None, force_overwrite=True):
        """Download pre-trained model."""
        download_path = (
            #os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            model_path
            if model_path is not None
            else os.path.join(WORKDIR, "models", "rhythm", "tcn-carnatic")
        )
        # Creating model folder to store the weights
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        download_remote_model(
            self.download_link,
            self.download_checksum,
            download_path,
            force_overwrite=force_overwrite,
        )

    def predict(self, audio_path):
        signal, sr = load_audio(audio_path)
        spect = self.signal2spect(signal, sr)
        beat_logits, downbeat_logits = self.spect2frames(spect)
        return self.frames2beats(beat_logits, downbeat_logits)

    @staticmethod
    def save_beats(data, output_path):
        """Calling the write_csv function in compiam.io to write the output beat track in a file

        :param data: the data to write
        :param output_path: the path where the data is going to be stored

        :returns: None
        """
        return write_csv(data, output_path)


    def select_gpu(self, gpu="-1"):
        """Select the GPU to use for inference.

        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        :returns: None
        """
        if int(gpu) == -1:
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:" + str(gpu))
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps:" + str(gpu))
            else:
                self.device = torch.device("cpu")
                logger.warning("No GPU available. Running on CPU.")
        self.gpu = gpu
