from typing import Any, List, Optional, Union
from typing_extensions import Literal
import torch
from torch import Tensor, quantile
from torchmetrics import Metric
from torchmetrics.classification import MultilabelAUROC, MultilabelAccuracy
from torchmetrics.wrappers import BootStrapper

class OracleModelCollaborativeAccuracy(MultilabelAccuracy):
    def __init__(self, alpha_in_percent:float, num_labels: int, threshold: float = 0.5, average: Literal['micro', 'macro', 'weighted', 'none'] | None = "macro", multidim_average: Literal['global', 'samplewise'] = "global", ignore_index: int | None = None, validate_args: bool = True, **kwargs: Any) -> None:
        super().__init__(num_labels, threshold, average, multidim_average, ignore_index, validate_args, **kwargs)
        self.alpha = alpha_in_percent /100
    
    def update(self, preds: Tensor, targets: Tensor):
        res = self._calculate_res(preds, targets)
        return super().update(preds, res)

    def _calculate_res(self, preds, targets):
        # gets the 1- alpha quantile of the predictions
        q1 = torch.quantile(preds, torch.Tensor([1-self.alpha]), dim=0, keepdim=False)
        # sets the value to 1 if pred in 1-alpha quantile
        u = self.calculate_u_score(preds)
        ones = torch.gt(u, q1).long()
        # gets the value if pred == target
        truth = torch.eq(torch.round(preds), targets)
        # truth = truth == False
        truth = truth.long()
        res = torch.where(ones == 1, ones, truth)
        return res
        
    def calculate_u_score(self, preds: torch.Tensor):
        # uncertainty = preds x (1 - preds)
        neg_preds = torch.subtract(torch.ones(preds.size()), preds)
        uncertainty_score = preds * neg_preds
        return uncertainty_score

class OracleModelCollaborativeAUROC(MultilabelAUROC):
    def __init__(self, alpha_in_percent: float, num_labels: int, average: Literal['micro', 'macro', 'weighted', 'none'] | None = "macro", thresholds: int | List[float] | Tensor | None = None, ignore_index: int | None = None, validate_args: bool = True):
        super().__init__(num_labels, average, thresholds, ignore_index, validate_args)
        self.alpha = alpha_in_percent /100

    def _calculate_res(self, preds, targets):
        # gets the 1- alpha quantile of the predictions
        q1 = torch.quantile(preds, torch.Tensor([1-self.alpha]), dim=0, keepdim=False)
        # sets the value to 1 if pred in 1-alpha quantile
        u = self.calculate_u_score(preds)
        ones = torch.gt(u, q1).long()
        # gets the value if pred == target
        truth = torch.eq(torch.round(preds), targets)
        # truth = truth == False
        truth = truth.long()
        res = torch.where(ones == 1, ones, truth)
        return res

    def calculate_u_score(self, preds: torch.Tensor):
        # uncertainty = preds x (1 - preds)
        neg_preds = torch.subtract(torch.ones(preds.size()), preds)
        uncertainty_score = preds * neg_preds
        return uncertainty_score

    def update(self, preds: Tensor, target: Tensor) -> None:
        res = self._calculate_res(preds, target)
        return super().update(preds, res)    