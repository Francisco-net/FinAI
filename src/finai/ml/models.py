"""Neural network models for financial predictions."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float = 0.2
    activation: str = "relu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class PositionalEncoding(nn.Module):
    """Positional encoding for time series data."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 252,  # ~1 ano de dias úteis
        output_dim: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)

        # Use last time step for prediction
        x = x[:, -1, :]
        return self.output_head(x)


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_out = lstm_out[:, -1, :]
        return self.output_head(last_out)


class GRUForecaster(nn.Module):
    """GRU model for time series forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_head = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.output_head(last_out)


class MLPPortfolioOptimizer(nn.Module):
    """MLP for portfolio optimization - predicts expected returns and risk."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Output heads
        self.shared = nn.Sequential(*layers)
        self.return_head = nn.Linear(prev_dim, config.output_dim)  # Expected returns
        self.risk_head = nn.Linear(prev_dim, config.output_dim)    # Risk (volatility)
        self.correlation_head = nn.Linear(prev_dim, config.output_dim * config.output_dim)  # Correlation matrix

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        expected_returns = self.return_head(features)
        risk = F.softplus(self.risk_head(features))  # Ensure positive
        correlation = torch.tanh(self.correlation_head(features))  # [-1, 1]
        return expected_returns, risk, correlation


class RiskAssessmentNet(nn.Module):
    """Neural network for risk assessment and profiling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64, 32],
        num_risk_classes: int = 3,  # conservador, moderado, arrojado
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.risk_classifier = nn.Linear(prev_dim, num_risk_classes)
        self.risk_score_head = nn.Linear(prev_dim, 1)  # Continuous risk score [0, 1]
        self.capacity_head = nn.Linear(prev_dim, 1)    # Risk capacity
        self.tolerance_head = nn.Linear(prev_dim, 1)   # Risk tolerance

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        return {
            "risk_class": self.risk_classifier(features),
            "risk_score": torch.sigmoid(self.risk_score_head(features)),
            "risk_capacity": torch.sigmoid(self.capacity_head(features)),
            "risk_tolerance": torch.sigmoid(self.tolerance_head(features)),
        }


class ReturnPredictor(nn.Module):
    """Neural network for asset return prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        num_assets: int = 10,
        prediction_horizon: int = 21,  # dias úteis (~1 mês)
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.prediction_horizon = prediction_horizon

        # Feature encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Prediction heads for each horizon
        self.horizon_heads = nn.ModuleList([
            nn.Linear(prev_dim, num_assets) for _ in range(prediction_horizon)
        ])

        # Uncertainty estimation
        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(prev_dim, num_assets) for _ in range(prediction_horizon)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)

        predictions = []
        uncertainties = []
        for i in range(self.prediction_horizon):
            pred = self.horizon_heads[i](features)
            unc = F.softplus(self.uncertainty_heads[i](features))
            predictions.append(pred)
            uncertainties.append(unc)

        # Stack: (batch_size, horizon, num_assets)
        predictions = torch.stack(predictions, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)

        return predictions, uncertainties


class MarketRegimeDetector(nn.Module):
    """Detects market regimes (bull, bear, sideways, volatile)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_regimes),
        )

        self.regime_probs = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_regimes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]

        logits = self.regime_classifier(last_out)
        probs = self.regime_probs(last_out)

        return logits, probs


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for robust predictions."""

    def __init__(self, models: list[nn.Module], weights: Optional[list[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]  # Take first output if tuple
            predictions.append(pred)

        # Weighted average
        stacked = torch.stack(predictions, dim=0)  # (num_models, batch, ...)
        weighted = stacked * self.weights.view(-1, 1, 1)
        return weighted.sum(dim=0)


# Model factory functions
def create_return_predictor(
    input_dim: int,
    num_assets: int = 8,
    prediction_horizon: int = 21,
    model_type: str = "transformer",
) -> nn.Module:
    """Factory for return prediction models."""
    if model_type == "transformer":
        return TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            output_dim=num_assets,
        )
    elif model_type == "lstm":
        return LSTMForecaster(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=num_assets,
        )
    elif model_type == "gru":
        return GRUForecaster(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=num_assets,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_portfolio_optimizer(input_dim: int, num_assets: int) -> MLPPortfolioOptimizer:
    """Factory for portfolio optimizer."""
    config = ModelConfig(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        output_dim=num_assets,
        dropout=0.2,
        activation="gelu",
    )
    return MLPPortfolioOptimizer(config)


def create_risk_assessment(input_dim: int) -> RiskAssessmentNet:
    """Factory for risk assessment model."""
    return RiskAssessmentNet(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_risk_classes=3,
        dropout=0.3,
    )


def create_market_regime_detector(input_dim: int) -> MarketRegimeDetector:
    """Factory for market regime detector."""
    return MarketRegimeDetector(
        input_dim=input_dim,
        hidden_dim=128,
        num_regimes=4,
        num_layers=2,
        dropout=0.2,
    )