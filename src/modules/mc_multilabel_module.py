from typing import Any
from .multilabel_module import MultilabelModule
from torch.nn import Dropout
import torch
import logging

class MCDropoutModel(MultilabelModule):
    def __init__(self, 
                 network, 
                 output_activation, 
                 loss, 
                 optimizer, 
                 lr_scheduler, 
                 metrics, 
                 logging_params, 
                 num_epochs, 
                 len_trainset, 
                 task, 
                 class_weights_loss, 
                 label_counts,
                 mc_iteration = None):
        super().__init__(network, output_activation, loss, optimizer, lr_scheduler, metrics, logging_params, num_epochs, len_trainset, task, class_weights_loss, label_counts)
        self.mc_iteration = mc_iteration
        self.dropout = Dropout()
        logging.info(f"Instantiating with dropout and iterations {mc_iteration}")
        self.model_uncertainty = []
        self.calibration_metrics = self.metrics["calibration"].clone("test/")
    
    # def test_model_step(self, batch, batch_idx):
    #     logits = self.predict_step(batch, batch_idx)
    #     loss = self.loss(logits, batch["labels"])
    #     preds = self.output_activation(logits)
    #     return loss, preds, batch["labels"]
    
    def predict_step(self, batch, batch_idx) -> Any:
        self.dropout.train()
        
        pred = [self.dropout(self.forward(**batch)).unsqueeze(0) for _ in range(self.mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        pred = self.output_activation(pred)
        return pred
    
    def test_step(self, batch, batch_idx):
        return_value = super().test_step(batch, batch_idx)
        model_unc = self.predict_step(batch, batch_idx)
        self.model_uncertainty.append(model_unc.detach().cpu())
        return return_value
    
    def on_test_epoch_end(self):
        test_targets = torch.cat(self.test_targets).int()
        test_preds = torch.cat(self.test_preds)
        test_muc = torch.cat(self.model_uncertainty)
        self.test_complete_metrics(test_preds, test_targets)
        self.calibration_metrics(test_preds, test_targets, test_muc)
        
        log_dict = {}

        # Rename cmap to cmap5!
        for metric_name, metric in self.test_complete_metrics.named_children():
            # Check for padding_factor attribute
            if hasattr(metric, 'sample_threshold') and metric.sample_threshold == 5:
                modified_name = 'cmAP5'
            else:
                modified_name = metric_name
            log_dict[f"test/{modified_name}"] = metric
        
        for metric_name, metric in self.calibration_metrics.named_children():
            log_dict[f"test/{metric_name}"] = metric

        self.log_dict(log_dict, **self.logging_params)
