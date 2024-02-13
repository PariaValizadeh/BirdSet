from src.modules.models.embedding_models.perch.inference import models
from torch import nn
from ml_collections import ConfigDict, config_dict
import numpy as np
import torch
import tf

class TfEmbeddingModel(nn.Module):
    def __init__(self, tf_model, embedding_dimension, num_classes, is_embedding:True):
        super().__init__()
        self.tf_model = tf_model
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes
        self.is_embedding = is_embedding
        if is_embedding:
            self.forward = self.forward_embed
        else:
            self.forward = self.forward_logits
    
    def forward_embed(self, input_values, device):
        input_values = input_values.cpu()
        inference = self.run_tf_model(input_values)
        embeddings = inference.embeddings
        embeddings = self.transform_embeddings(embeddings)
        if device is None:
            return torch.from_numpy(embeddings)
        return torch.from_numpy(embeddings).to(device)
    
    @tf.function
    def run_tf_model(self, input: tf.tensor):
        inference = self.tf_model(input)
        return inference
    
    def forward_logits(self, input_values, device):
        input_values = input_values.cpu()
        inference = self.run_tf_model(input_values)
        logits = inference.logits
        logits = self.transform_logits(logits)
        logits = torch.from_numpy(logits).to(device)
        return logits
    
    def transform_logits(logits):
        return logits
    
    def transform_embeddings(self, embeddings):
        return embeddings

class DownloadTfEmbeddingModel(TfEmbeddingModel):
    def __init__(self, config_dict:config_dict.ConfigDict, model_key:str, is_embedding=True):
        model_class = models.model_class_map()[model_key]
        tf_wrapper = model_class.from_config(config_dict)
        tf_model = tf_wrapper.batch_embed
        super().__init__(tf_model, embedding_dimension=config_dict.embed_dim, num_classes=config_dict.num_classes, is_embedding=is_embedding)

class PerchTfEmbeddingModel(DownloadTfEmbeddingModel):
    def __init__(self, 
                 is_embedding = True,
                 window_size_s:float = 5.0, 
                 hop_size_s:float = 5.0, 
                 sample_rate:int = 32000, 
                 num_classes:int = 10932, 
                 embed_dim:int = 1280, 
                 model_path:str = "", 
                 tfhub_version: int = 4,
                 model_key:str = "taxonomy_model_tf"):
        self.config = config_dict.ConfigDict()
        self.config.window_size_s = window_size_s
        self.config.hop_size_s = hop_size_s
        self.config.sample_rate = sample_rate
        self.config.num_classes = num_classes
        self.config.embed_dim = embed_dim
        self.config.model_path = model_path
        self.config.tfhub_version = tfhub_version
        super().__init__(config_dict=self.config, model_key=model_key, is_embedding=is_embedding)

class BirdNetTfEmbeddingModel(DownloadTfEmbeddingModel):
    def __init__(self, 
                 is_embedding=True,
                 window_size_s:float = 3.0, 
                 hop_size_s:float = 3.0, 
                 sample_rate:int = 48000, 
                 num_classes:int = 6522, 
                 embed_dim:int = 1024, 
                 model_path:str = "/Users/moritzrichert/Downloads/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite", 
                 num_tflite_threads:int = 4,
                 class_list_name:str = "birdnet_v2_4",
                 model_key:str = "birdnet"):
        self.config = config_dict.ConfigDict()
        self.config.window_size_s = window_size_s
        self.config.hop_size_s = hop_size_s
        self.config.sample_rate = sample_rate
        self.config.num_classes = num_classes
        self.config.embed_dim = embed_dim
        self.config.model_path = model_path
        self.config.num_tflite_threads = num_tflite_threads
        self.config.class_list_name = class_list_name
        super().__init__(config_dict=self.config, model_key=model_key, is_embedding=is_embedding)

class AverageTfEmbeddingModel(DownloadTfEmbeddingModel):
    def __init__(self, config_dict:config_dict.ConfigDict, model_key:str, is_embedding = True):
        super().__init__(config_dict, model_key, is_embedding=is_embedding)
    
    def transform_embeddings(self, embeddings):
        embeddings = embeddings.mean(axis=1)
        return super().transform_embeddings(embeddings)

class YamnetTfEmbeddingModel(AverageTfEmbeddingModel):
    def __init__(self, 
                 is_embedding = True,
                 window_size_s:float = 3.0,
                 hop_size_s:float = 3.0,
                 sample_rate:int = 16000,
                 num_classes:int = 521,
                 embed_dim:int = 1024,
                 model_path:str = "https://tfhub.dev/google/yamnet/1",
                 embedding_index:int = 1,
                 logits_index:int = 0,
                 model_key:str = "tfhub_model"
                 ):
        self.config = config_dict.ConfigDict()
        self.config.window_size_s = window_size_s
        self.config.hop_size_s = hop_size_s
        self.config.sample_rate = sample_rate
        self.config.num_classes = num_classes
        self.config.embed_dim = embed_dim
        self.config.model_url = model_path
        self.config.embedding_index = embedding_index
        self.config.logits_index = logits_index
        super().__init__(config_dict=self.config, model_key=model_key, is_embedding=is_embedding)

class VGGishTfEmbeddingModel(AverageTfEmbeddingModel):
    def __init__(self, 
                 is_embedding = True,
                 window_size_s:float = 3.0,
                 hop_size_s:float = 3.0,
                 sample_rate:int = 16000,
                 num_classes:int = 128,
                 embed_dim:int = 128,
                 model_path:str = "https://tfhub.dev/google/vggish/1",
                 embedding_index:int = -1,
                 logits_index:int = -1,
                 model_key:str = "tfhub_model"
                 ):
        self.config = config_dict.ConfigDict()
        self.config.window_size_s = window_size_s
        self.config.hop_size_s = hop_size_s
        self.config.sample_rate = sample_rate
        self.config.num_classes = num_classes
        self.config.embed_dim = embed_dim
        self.config.model_url = model_path
        self.config.embedding_index = embedding_index
        self.config.logits_index = logits_index
        super().__init__(config_dict=self.config, model_key=model_key, is_embedding=is_embedding)