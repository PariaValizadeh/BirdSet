from typing import Any
import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC

class CalibrationAuroc(Metric):
    def __init__(self, **kwargs: Any) -> None:
        # see here:
        # https://github.com/google/uncertainty-baselines/blob/master/baselines/toxic_comments/metrics.py
        #  Calibration AUC
        # and here:
        # https://papers.nips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf
        super().__init__(**kwargs)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> Any:
        assert preds.shape == target.shape
        
        

    def compute(self):
        pass
    

def calculate(preds: torch.Tensor, target: torch.Tensor):
    ones = torch.ones(preds.size())
    neg_preds = ones - preds
    # uncertainty = preds x (1 - preds)
    uncertainty_score = torch.matmul(preds, neg_preds)

def main():
    target = [0, 1, 0, 1, 0]
    preds = [0, 0.78, 0, 0.5, 0]
    
    cal = CalibrationAuroc()
    cal(torch.tensor(preds), torch.tensor(target))

if __name__ == "__main__":
    main()