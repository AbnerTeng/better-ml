"""
Gradient Boosting Trainer
"""
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn_best_practice.base import BaseModel


class RFTrainer(BaseModel):
    """
    Random Forest Trainer
    """

    def __init__(
        self,
        model: RandomForestClassifier,
        config: None | Dict[str, Any]
    ) -> None:
        super().__init__(model, config)


class XGBTrainer(BaseModel):
    """
    XGBoost Trainer
    """

    def __init__(
        self,
        model: XGBClassifier,
        config: None | Dict[str, Any]
    ) -> None:
        super().__init__(model, config)


class LGBMTrainer(BaseModel):
    """
    LightGBM Trainer
    """

    def __init__(
        self,
        model: LGBMClassifier,
        config: None | Dict[str, Any]
    ) -> None:
        super().__init__(model, config)
