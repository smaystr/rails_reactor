import torch

from models.base_model import BaseModel, BaseTorchModel


class LinearReg(BaseModel):
    def score(self, X: torch.Tensor, y: torch.Tensor, metric_type: str = "mse") -> float:
        return BaseModel.score(self, X, y, metric_type)


class TorchLinearReg(BaseTorchModel):
    def forward(self, X):
        out = self.linear(X)
        return out
