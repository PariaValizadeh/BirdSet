from typing import Sequence
import numpy as np
from src.perch.inference import interface


def model_outputs_to_tf_example(
    model_outputs: interface.InferenceOutputs,
    file_id: str,
    audio: np.ndarray,
    timestamp_offset_s: float,
    write_embeddings: bool,
    write_logits: bool | Sequence[str],
    write_separated_audio: bool,
    write_raw_audio: bool,
):
    print(f"Handling {file_id}")
    pass