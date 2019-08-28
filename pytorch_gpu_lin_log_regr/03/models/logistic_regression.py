import torch

from models.base_model import BaseModel, BaseTorchModel


class LogisticReg(BaseModel):
    def score(self, X: torch.Tensor, y: torch.Tensor, metric_type: str = "accuracy") -> float:
        return BaseModel.score(self, X, y, metric_type)

    @staticmethod
    def _decision_function(X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return LogisticReg._sigmoid(BaseModel._decision_function(X, theta)).round().float()

    @staticmethod
    def _sigmoid(z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-z))


class TorchLogisticReg(BaseTorchModel):
    def forward(self, X):
        out = torch.sigmoid(self.linear(X))
        return out
