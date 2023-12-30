from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn, optim
from src.modules.models.embedding_models.perch_tf_embedding_model import TfEmbeddingModel
import torchmetrics
import torch
import hydra

class EmbeddingClassifier(nn.Module):
    def __init__(self, tf_model, num_classes, device) -> None:
        super().__init__()
        self.embedding_model = tf_model
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features=self.embedding_model.embedding_dimension, out_features=num_classes)
        self.device = device
        # this is from the feature embeddings paper
        # using cce does not seem to be natively supported by torch
        # self.m = nn.Sigmoid() # this can be configured via module
        # self.criterion = nn.BCELoss()
        # bceloss is still broken, falling back to crossentropy
        # self.criterion = nn.CrossEntropyLoss()
        # different erros
        # self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        # self.ece = torchmetrics.CalibrationError(task="multiclass", num_classes=num_classes)
        # self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
    
    def forward(self, input_values, return_hidden_state=False, **kwargs) -> Any:
        input_values = input_values.cpu()
        embeddings = self.embedding_model(input_values)
        embeddings = torch.from_numpy(embeddings).to(self.device)
        logits = self.linear(embeddings)
        logits = logits.unsqueeze(0)
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
        
    # def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
    #     x = batch["input_values"]
    #     y = batch["labels"]
    #     x = x.cpu()
    #     result = self(x)
        
    #     loss = self.criterion(result, y)
    #     acc = self.acc(result, y)
    #     ece = self.ece(result, y)
    #     auroc = self.auroc(result, y)
    #     self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_ece", ece, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_auroc", auroc, on_epoch=True, prog_bar=True, logger=True)
    #     return loss
    
    # def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
    #     x = batch["input_values"]
    #     y = batch["labels"]
    #     x = x.cpu()
    #     pass
        
    # def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
    #     x = batch["input_values"]
    #     y = batch["labels"]
    #     x = x.cpu()
    #     pass
    
    # def configure_optimizers(self) -> OptimizerLRScheduler:
    #     optimizer = optim.Adam(self.parameters(), self.learning_rate)
    #     return optimizer