from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn, optim
from src.modules.models.embedding_models.perch_tf_embedding_model import TfEmbeddingModel
import torchmetrics
import torch
import hydra

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_model, num_classes, device) -> None:
        # device is necessary as the embedding model is most likely to run on the cpu (due to tensorflow integration)
        # -> numpy arrays are passed to the cpu, fed into tensorflow, and afterwards transferred to the gpu again
        super().__init__()
        if device == "gpu":
            device = "cuda"
        self.embedding_model = embedding_model
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features=self.embedding_model.embedding_dimension, out_features=num_classes)
        self.device = device
    
    def forward(self, input_values, **kwargs) -> Any:
        embeddings = self.embedding_model(input_values, self.device)
        logits = self.linear(embeddings)
        # logits = logits.unsqueeze(0)
        return logits
    
    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass 