from typing import Any, List, Optional, Union
from typing_extensions import Literal
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MultilabelAUROC


class RejectionAccuracy(Metric):
    def __init__(self, rejection_rate_in_percent: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rejection_rate = rejection_rate_in_percent /100
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, targets: Tensor):
        self.preds.append(preds)
        self.targets.append(targets)
    
    def calculate_u_score(self, preds: torch.Tensor):
        # uncertainty = preds x (1 - preds)
        neg_preds = torch.subtract(torch.ones(preds.size()), preds)
        uncertainty_score = preds * neg_preds
        return uncertainty_score
        
    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.preds).int()
        u = self.calculate_u_score(preds)
        u = torch.mean(u, dim=1, keepdim=True)
        l = len(u)
        k = int((1- self.rejection_rate) * l)
        indices = (torch.topk(u, k, dim=0)[1]).squeeze()
        print(indices.shape)
        top_preds = torch.index_select(preds, 0, indices.squeeze())
        top_targets = torch.index_select(targets, 0, indices.squeeze())
        res = torch.eq(torch.round(top_preds), top_targets).long()
        return torch.mean(res, dtype=torch.float)

class RejectionAUROC(MultilabelAUROC):
    def __init__(self, rejection_rate_in_percent:float, num_labels: int, average: Literal['micro', 'macro', 'weighted', 'none'] | None = "macro", thresholds: int | List[float] | Tensor | None = None, ignore_index: int | None = None, validate_args: bool = True, **kwargs: Any) -> None:
        super().__init__(num_labels, average, thresholds, ignore_index, validate_args, **kwargs)
        self.rejection_rate = rejection_rate_in_percent /100
    
    def calculate_u_score(self, preds: torch.Tensor):
        # uncertainty = preds x (1 - preds)
        neg_preds = torch.subtract(torch.ones(preds.size()), preds)
        uncertainty_score = preds * neg_preds
        return uncertainty_score
    
    def compute(self) -> Tensor:
        preds = torch.cat(self.preds)
        targets = torch.cat(self.preds).int()
        u = self.calculate_u_score(preds)
        u = torch.mean(u, dim=1, keepdim=True)
        l = len(u)
        k = int((1- self.rejection_rate) * l)
        indices = (torch.topk(u, k, dim=0)[1]).squeeze()
        print(indices.shape)
        top_preds = torch.index_select(preds, 0, indices.squeeze())
        top_targets = torch.index_select(targets, 0, indices.squeeze())
        self.preds = top_preds.tolist()
        self.targets = top_targets.tolist()
        return super().compute()
    
            
        
        