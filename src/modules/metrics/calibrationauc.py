from typing import Any, List, Literal, Optional, Union
from typing_extensions import Literal
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MultilabelAUROC

class CalibrationAuroc(BinaryAUROC):
    def __init__(self, 
                 mode: Literal["multilabel", "multiclass"] = "multiclass",
                 **kwargs: Any) -> None:
        # see here:
        # https://github.com/google/uncertainty-baselines/blob/master/baselines/toxic_comments/metrics.py
        #  Calibration AUC
        # and here:
        # https://papers.nips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf
        super().__init__(**kwargs)
        if mode == "multiclass":
            self.eq_calc = self.multiclass_eq_calc
        elif mode == "multilabel":
            raise ValueError("Multilabel is unsupported!")
            self.eq_calc = self.multilabel_eq_calc
            task = "multilabel"
        else:
            raise ValueError(f"There was something wrong with {mode}")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> Any:
        if preds.shape != target.shape:
            print(f"Preds: {preds.shape}, target: {target.shape}")
        assert preds.shape == target.shape
        u_score = self.calculate_u_score(preds)
        eq = self.multiclass_eq_calc(preds, target)
        return super().update(u_score, eq)
    
    def multiclass_eq_calc(self, preds, target):
        # this creates the "hard" predictions
        one_preds = torch.round(preds)
        
        inv_eq = torch.eq(one_preds, target)
        eq = inv_eq == False
        eq.long()
        return eq
    
    # def multilabel_eq_calc(self, preds, target):
    #     one_preds = torch.prod(preds, 1, True)
    #     eq = torch.eq(one_preds, target).long()
    #     return eq
        
        
    def calculate_u_score(self, preds: torch.Tensor):
        # uncertainty = preds x (1 - preds)
        neg_preds = torch.subtract(torch.ones(preds.size()), preds)
        uncertainty_score = preds * neg_preds
        return uncertainty_score

class MultilabelCalibrationAuroc(MultilabelAUROC):
    def __init__(self, num_labels: int, average: Literal['micro', 'macro', 'weighted', 'none'] | None = "macro", thresholds: int | List[float] | Tensor | None = None, ignore_index: int | None = None, validate_args: bool = True, **kwargs: Any) -> None:
        super().__init__(num_labels, average, thresholds, ignore_index, validate_args, **kwargs)
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        u = self.calculate_u_score(preds)
        return super().update(preds, target)
    
    def calculate_u_score(self, preds: torch.Tensor):
        # uncertainty = preds x (1 - preds)
        neg_preds = torch.subtract(torch.ones(preds.size()), preds)
        uncertainty_score = preds * neg_preds
        return uncertainty_score

def main():
    target = [[0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1]]
    preds = [[0.01, 0.2, 0.55], [0.78, 0.51, 0.1], [0.01, 0.4, 0.55], [0.9, 0.1, 0.7], [0.1, 0.6, 0.4]]
    preds = torch.tensor(preds)
    target = torch.tensor(target)
    
    cal = CalibrationAuroc(mode="multilabel")
    print(cal(preds, target))
    print(cal(preds, target))

if __name__ == "__main__":
    main()