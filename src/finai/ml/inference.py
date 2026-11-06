"""Inference utilities for ML models in production."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from finai.ml.models import (
    ReturnPredictor,
    RiskAssessmentNet,
    MarketRegimeDetector,
    MLPPortfolioOptimizer,
    create_return_predictor,
    create_portfolio_optimizer,
    create_risk_assessment,
    create_market_regime_detector,
)
from finai.ml.data import FinancialFeatureEngineer, FeatureConfig, prepare_risk_features

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container for model artifacts."""
    model: nn.Module
    feature_engineer: Optional[FinancialFeatureEngineer]
    metadata: dict[str, Any]
    version: str


class ModelRegistry:
    """Registry for managing ML models."""

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, ModelArtifacts] = {}

    def register(
        self,
        name: str,
        model: nn.Module,
        feature_engineer: Optional[FinancialFeatureEngineer] = None,
        metadata: Optional[dict] = None,
        version: str = "1.0.0",
    ) -> None:
        """Register a model."""
        self._models[name] = ModelArtifacts(
            model=model,
            feature_engineer=feature_engineer,
            metadata=metadata or {},
            version=version,
        )
        logger.info(f"Registered model: {name} v{version}")

    def get(self, name: str) -> Optional[ModelArtifacts]:
        """Get a registered model."""
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self._models.keys())

    def save_model(self, name: str, path: Optional[str] = None) -> str:
        """Save model to disk."""
        if name not in self._models:
            raise ValueError(f"Model {name} not registered")

        artifacts = self._models[name]
        if path is None:
            path = str(self.model_dir / f"{name}_v{artifacts.version}.pt")

        save_dict = {
            'model_state_dict': artifacts.model.state_dict(),
            'model_class': artifacts.model.__class__.__name__,
            'metadata': artifacts.metadata,
            'version': artifacts.version,
        }

        # Save feature engineer config if exists
        if artifacts.feature_engineer:
            save_dict['feature_config'] = artifacts.feature_engineer.config.__dict__

        torch.save(save_dict, path)
        logger.info(f"Saved model {name} to {path}")
        return path

    def load_model(
        self,
        name: str,
        path: str,
        model_class: type,
        feature_config: Optional[FeatureConfig] = None,
        **model_kwargs,
    ) -> ModelArtifacts:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location="cpu")

        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        feature_engineer = None
        if 'feature_config' in checkpoint and feature_config is None:
            feature_config = FeatureConfig(**checkpoint['feature_config'])

        if feature_config:
            feature_engineer = FinancialFeatureEngineer(feature_config)
            # Note: scaler would need to be fitted separately or saved

        artifacts = ModelArtifacts(
            model=model,
            feature_engineer=feature_engineer,
            metadata=checkpoint.get('metadata', {}),
            version=checkpoint.get('version', '1.0.0'),
        )

        self._models[name] = artifacts
        logger.info(f"Loaded model {name} from {path}")
        return artifacts


class InferenceEngine:
    """High-level inference engine for FinAI ML models."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_returns(
        self,
        model_name: str,
        price_data: np.ndarray,  # (seq_len, n_features)
        return_uncertainty: bool = True,
    ) -> dict[str, np.ndarray]:
        """Predict future returns."""
        artifacts = self.registry.get(model_name)
        if artifacts is None:
            raise ValueError(f"Model {model_name} not found")

        model = artifacts.model.to(self.device)
        model.eval()

        # Prepare input
        x = torch.FloatTensor(price_data).unsqueeze(0).to(self.device)  # Add batch dim

        with torch.no_grad():
            outputs = model(x)

        if isinstance(outputs, tuple):
            predictions, uncertainties = outputs
            return {
                "predictions": predictions.cpu().numpy().squeeze(0),
                "uncertainties": uncertainties.cpu().numpy().squeeze(0) if return_uncertainty else None,
            }
        else:
            return {
                "predictions": outputs.cpu().numpy().squeeze(0),
                "uncertainties": None,
            }

    def assess_risk(
        self,
        model_name: str,
        profile_data: dict,
        market_data: Optional[dict] = None,
    ) -> dict[str, float]:
        """Assess investor risk profile."""
        artifacts = self.registry.get(model_name)
        if artifacts is None:
            raise ValueError(f"Model {model_name} not found")

        model = artifacts.model.to(self.device)
        model.eval()

        # Prepare features
        features = prepare_risk_features(profile_data, market_data)
        x = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            outputs = model(x)

        return {
            "risk_class_probs": torch.softmax(outputs["risk_class"], dim=-1).cpu().numpy().squeeze(),
            "risk_score": outputs["risk_score"].cpu().item(),
            "risk_capacity": outputs["risk_capacity"].cpu().item(),
            "risk_tolerance": outputs["risk_tolerance"].cpu().item(),
        }

    def detect_market_regime(
        self,
        model_name: str,
        market_features: np.ndarray,  # (seq_len, n_features)
    ) -> dict[str, np.ndarray]:
        """Detect current market regime."""
        artifacts = self.registry.get(model_name)
        if artifacts is None:
            raise ValueError(f"Model {model_name} not found")

        model = artifacts.model.to(self.device)
        model.eval()

        x = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, probs = model(x)

        regime_names = ["bull", "bear", "sideways", "volatile"]
        return {
            "regime_logits": logits.cpu().numpy().squeeze(),
            "regime_probs": probs.cpu().numpy().squeeze(),
            "regime_names": regime_names,
            "most_likely": regime_names[probs.argmax().item()],
        }

    def optimize_portfolio(
        self,
        model_name: str,
        market_features: np.ndarray,  # (n_features,)
    ) -> dict[str, np.ndarray]:
        """Get portfolio optimization predictions."""
        artifacts = self.registry.get(model_name)
        if artifacts is None:
            raise ValueError(f"Model {model_name} not found")

        model = artifacts.model.to(self.device)
        model.eval()

        x = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            expected_returns, risk, correlation = model(x)

        n_assets = expected_returns.shape[-1]
        correlation_matrix = correlation.view(n_assets, n_assets).cpu().numpy()

        # Make correlation matrix symmetric and positive definite
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)

        return {
            "expected_returns": expected_returns.cpu().numpy().squeeze(),
            "risk": risk.cpu().numpy().squeeze(),
            "correlation_matrix": correlation_matrix,
        }


def create_default_models(
    n_assets: int = 8,
    n_features: int = 50,
    sequence_length: int = 60,
) -> dict[str, nn.Module]:
    """Create default models for FinAI."""
    return {
        "return_predictor": create_return_predictor(
            input_dim=n_features,
            num_assets=n_assets,
            prediction_horizon=21,
            model_type="transformer",
        ),
        "risk_assessment": create_risk_assessment(input_dim=20),
        "market_regime": create_market_regime_detector(input_dim=n_features),
        "portfolio_optimizer": create_portfolio_optimizer(input_dim=n_features, num_assets=n_assets),
    }


def initialize_inference_engine(
    model_dir: str = "./models",
    n_assets: int = 8,
    n_features: int = 50,
) -> InferenceEngine:
    """Initialize inference engine with default models."""
    registry = ModelRegistry(model_dir)
    engine = InferenceEngine(registry)

    models = create_default_models(n_assets, n_features)

    for name, model in models.items():
        engine.registry.register(
            name=name,
            model=model,
            metadata={"n_assets": n_assets, "n_features": n_features},
            version="1.0.0",
        )

    return engine