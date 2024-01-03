# coding=utf-8
# Copyright 2023 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations of inference interfaces for applying trained models."""

import dataclasses
import tempfile
from typing import Any

from absl import logging
from src.modules.models.embedding_models.perch.inference import interface
from src.modules.models.embedding_models.perch.taxonomy import namespace
from src.modules.models.embedding_models.perch.taxonomy import namespace_db
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

PERCH_TF_HUB_URL = 'https://tfhub.dev/google/bird-vocalization-classifier'


def model_class_map() -> dict[str, Any]:
  """Get the mapping of model keys to classes."""
  return {
      'taxonomy_model_tf': TaxonomyModelTF,
      'birdnet': BirdNet,
      'tfhub_model': TFHubModel,
  }

@dataclasses.dataclass
class TaxonomyModelTF(interface.EmbeddingModel):
  """Taxonomy SavedModel.

  Attributes:
    model_path: Path to model files.
    window_size_s: Window size for framing audio in seconds. TODO(tomdenton):
      Ideally this should come from a model metadata file.
    hop_size_s: Hop size for inference.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
    batchable: Whether the model supports batched input.
    target_peak: Peak normalization value.
  """

  model_path: str
  window_size_s: float
  hop_size_s: float
  model: Any  # TF SavedModel
  class_list: namespace.ClassList
  batchable: bool
  target_peak: float | None = 0.25
  tfhub_version: int | None = None
  num_classes:int |None = None
  embed_dim:int |None = None

  @classmethod
  def is_batchable(cls, model: Any) -> bool:
    sig = model.signatures['serving_default']
    return sig.inputs[0].shape[0] is None

  @classmethod
  def from_tfhub(cls, config: config_dict.ConfigDict) -> 'TaxonomyModelTF':
    if not hasattr(config, 'tfhub_version') or config.tfhub_version is None:
      raise ValueError('tfhub_version is required to load from TFHub.')
    if config.model_path:
      raise ValueError(
          'Exactly one of tfhub_version and model_path should be set.'
      )

    model_url = f'{PERCH_TF_HUB_URL}/{config.tfhub_version}'
    # This model behaves exactly like the usual saved_model.
    model = hub.load(model_url)

    # Check whether the model support polymorphic batch shape.
    batchable = cls.is_batchable(model)

    # Get the labels CSV from TFHub.
    model_path = hub.resolve(model_url)
    labels_path = epath.Path(model_path) / 'assets/label.csv'
    with labels_path.open('r') as f:
      class_list = namespace.ClassList.from_csv(f)
    return cls(
        model=model, class_list=class_list, batchable=batchable, **config
    )

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'TaxonomyModelTF':
    logging.info('Loading taxonomy model...')

    if hasattr(config, 'tfhub_version') and config.tfhub_version is not None:
      return cls.from_tfhub(config)

    base_path = epath.Path(config.model_path)
    if (base_path / 'saved_model.pb').exists() and (
        base_path / 'assets'
    ).exists():
      # This looks like a downloaded TFHub model.
      model_path = base_path
      label_csv_path = epath.Path(config.model_path) / 'assets' / 'label.csv'
    else:
      # Probably a savedmodel distributed directly.
      model_path = base_path / 'savedmodel'
      label_csv_path = base_path / 'label.csv'

    model = tf.saved_model.load(model_path)
    with label_csv_path.open('r') as f:
      class_list = namespace.ClassList.from_csv(f)

    # Check whether the model support polymorphic batch shape.
    batchable = cls.is_batchable(model)
    return cls(
        model=model, class_list=class_list, batchable=batchable, **config
    )

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    if self.batchable:
      return interface.embed_from_batch_embed_fn(self.batch_embed, audio_array)

    # Process one example at a time.
    # This should be fine on CPU, but may be somewhat inefficient for large
    # arrays on GPU or TPU.
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    if self.target_peak is not None:
      framed_audio = self.normalize_audio(framed_audio, self.target_peak)
    all_logits, all_embeddings = self.model.infer_tf(framed_audio[:1])
    for window in framed_audio[1:]:
      logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    # it is likely that the following line has to be removed
    all_embeddings = all_embeddings[:, np.newaxis, :]
    print("All embeddings")
    print(all_embeddings.shape)
    return interface.InferenceOutputs(
        all_embeddings, {'label': all_logits}, None
    )

  def batch_embed(
      self, audio_batch: np.ndarray[Any, Any]
  ) -> interface.InferenceOutputs:
    if not self.batchable:
      return interface.batch_embed_from_embed_fn(self.embed, audio_batch)

    framed_audio = self.frame_audio(
        audio_batch, self.window_size_s, self.hop_size_s
    )
    if self.target_peak is not None:
      framed_audio = self.normalize_audio(framed_audio, self.target_peak)

    rebatched_audio = framed_audio.reshape([-1, framed_audio.shape[-1]])
    logits, embeddings = self.model.infer_tf(rebatched_audio)
    logits = np.reshape(logits, framed_audio.shape[:2] + (logits.shape[-1],))
    # embeddings = np.reshape(
    #     embeddings, framed_audio.shape[:2] + (embeddings.shape[-1],)
    # )
    # embeddings = embeddings.squeeze()
    # print(embeddings.shape)
    return interface.InferenceOutputs(embeddings, {'label': logits}, None)


@dataclasses.dataclass
class BirdNet(interface.EmbeddingModel):
  """Wrapper for BirdNet models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
    window_size_s: Window size for framing audio in samples.
    hop_size_s: Hop size for inference.
    num_tflite_threads: Number of threads to use with TFLite model.
    class_list_name: Name of the BirdNet class list.
  """

  model_path: str
  model: Any
  tflite: bool
  class_list: namespace.ClassList
  window_size_s: float = 3.0
  hop_size_s: float = 3.0
  num_tflite_threads: int = 16
  class_list_name: str = 'birdnet_v2_1'
  num_classes:int |None = None
  embed_dim:int |None = None

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'BirdNet':
    logging.info('Loading BirdNet model...')
    if config.model_path.endswith('.tflite'):
      tflite = True
      with tempfile.NamedTemporaryFile() as tmpf:
        model_file = epath.Path(config.model_path)
        model_file.copy(tmpf.name, overwrite=True)
        model = tf.lite.Interpreter(
            tmpf.name, num_threads=config.num_tflite_threads
        )
      model.allocate_tensors()
    else:
      tflite = False
      model = tf.saved_model.load(config.model_path)
    db = namespace_db.load_db()
    class_list = db.class_lists[config.class_list_name]
    return cls(
        model=model,
        tflite=tflite,
        class_list=class_list,
        **config,
    )

  def embed_saved_model(
      self, audio_array: np.ndarray
  ) -> interface.InferenceOutputs:
    """Get logits using the BirdNet SavedModel."""
    # Note that there is no easy way to get the embedding from the SavedModel.
    all_logits = self.model(audio_array[:1])
    for window in audio_array[1:]:
      logits = self.model(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
    return interface.InferenceOutputs(
        None, {self.class_list_name: all_logits}, None
    )

  def embed_tflite(self, audio_array: np.ndarray, reshape:bool=True) -> interface.InferenceOutputs:
    """Create an embedding and logits using the BirdNet TFLite model."""
    # reshape needs to be set to true when augmentations from transforms are used... (which is not correct.)
    input_details = self.model.get_input_details()[0]
    output_details = self.model.get_output_details()[0]
    embedding_idx = output_details['index'] - 1
    embeddings = []
    logits = []
    for audio in audio_array:
      input = np.float32(audio)
      if reshape:
        input = input[np.newaxis, :]
      self.model.set_tensor(
          input_details['index'], input
      )
      self.model.invoke()
      logits.append(self.model.get_tensor(output_details['index']))
      embeddings.append(self.model.get_tensor(embedding_idx))
    # Create [Batch, 1, Features]
    embeddings = np.array(embeddings)
    logits = np.array(logits)
    embeddings = embeddings.squeeze()
    return interface.InferenceOutputs(
        embeddings, {self.class_list_name: logits}, None
    )

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    if self.tflite:
      return self.embed_tflite(framed_audio)
    else:
      return self.embed_saved_model(framed_audio)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class TFHubModel(interface.EmbeddingModel):
  """Generic wrapper for TFHub models which produce embeddings."""

  model: Any  # TFHub loaded model.
  model_url: str
  embedding_index: int
  logits_index: int = -1
  window_size_s: float | None = None
  hop_size_s:float | None = None
  num_classes:int |None = None
  embed_dim:int |None = None


  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'TFHubModel':
    model = hub.load(config.model_url)
    return cls(
        model=model,
        **config,
    )
  
  @classmethod
  def yamnet_cfg(cls):
    config = config_dict.ConfigDict({
        'sample_rate': 16000,
        'model_url': 'https://tfhub.dev/google/yamnet/1',
        'embedding_index': 1,
        'logits_index': 0,
    })
    return config
  
  @classmethod
  def yamnet(cls):
    # Parent class takes a sample_rate arg which pylint doesn't find.
    config = TFHubModel.yamnet_cfg()
    return TFHubModel.from_config(config)
  
  @classmethod
  def vggish_cfg(cls):
    config = config_dict.ConfigDict({
        'sample_rate': 16000,
        'model_url': 'https://tfhub.dev/google/vggish/1',
        'embedding_index': -1,
        'logits_index': -1,
    })
    return config
  
  @classmethod
  def vggish(cls):
    config = TFHubModel.vggish_cfg()
    return TFHubModel.from_config(config)

  def embed(
      self, audio_array: np.ndarray[Any, np.dtype[Any]]
  ) -> interface.InferenceOutputs:
    outputs = self.model(audio_array)
    if self.embedding_index < 0:
      embeddings = outputs
    else:
      embeddings = outputs[self.embedding_index]
    if len(embeddings.shape) == 1:
      embeddings = embeddings[np.newaxis, :]
    elif len(embeddings.shape) != 2:
      raise ValueError('Embeddings should have shape [Depth] or [Time, Depth].')

    if self.logits_index >= 0:
      logits = {'label': outputs[self.logits_index]}
    else:
      logits = None
    return interface.InferenceOutputs(embeddings, logits, None, False)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)
