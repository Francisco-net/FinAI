"""Data preprocessing and feature engineering for ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    lookback_window: int = 60  # days
    prediction_horizon: int = 21  # days
    use_technical_indicators: bool = True
    use_macro_features: bool = True
    use_volume_features: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust
    fill_missing: str = "forward"  # forward, backward, interpolate, drop


class FinancialFeatureEngineer:
    """Feature engineering for financial time series."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.fitted = False

    def _get_scaler(self):
        """Get scaler based on config."""
        if self.config.scaling_method == "standard":
            return StandardScaler()
        elif self.config.scaling_method == "minmax":
            return MinMaxScaler()
        elif self.config.scaling_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")

    def add_technical_indicators(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Add technical indicators to price data."""
        df = df.copy()

        # Returns
        df["returns"] = df[price_col].pct_change()
        df["log_returns"] = np.log(df[price_col] / df[price_col].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f"sma_{window}"] = df[price_col].rolling(window).mean()
            df[f"ema_{window}"] = df[price_col].ewm(span=window).mean()

        # Bollinger Bands
        df["bb_middle"] = df[price_col].rolling(20).mean()
        bb_std = df[price_col].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df[price_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df[price_col].ewm(span=12).mean()
        exp2 = df[price_col].ewm(span=26).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Volatility
        df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
        df["volatility_60"] = df["returns"].rolling(60).std() * np.sqrt(252)

        # Momentum
        for window in [5, 10, 20, 60]:
            df[f"momentum_{window}"] = df[price_col] / df[price_col].shift(window) - 1

        # Volume indicators (if available)
        if "volume" in df.columns and self.config.use_volume_features:
            df["volume_sma_20"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
            df["obv"] = (np.sign(df["returns"]) * df["volume"]).cumsum()

        return df

    def add_macro_features(self, df: pd.DataFrame, macro_data: dict[str, pd.Series]) -> pd.DataFrame:
        """Add macroeconomic features."""
        df = df.copy()

        for name, series in macro_data.items():
            # Align macro data to price data dates
            aligned = series.reindex(df.index, method="ffill")
            df[f"macro_{name}"] = aligned
            df[f"macro_{name}_change"] = aligned.pct_change()

        return df

    def create_sequences(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for time series models."""
        X = []
        y = [] if targets is not None else None

        for i in range(len(features) - self.config.lookback_window - self.config.prediction_horizon + 1):
            X.append(features[i:i + self.config.lookback_window])

            if targets is not None:
                y.append(targets[i + self.config.lookback_window:
                                 i + self.config.lookback_window + self.config.prediction_horizon])

        X = np.array(X)
        if y is not None:
            y = np.array(y)

        return X, y

    def fit_transform(self, df: pd.DataFrame, target_cols: Optional[list[str]] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit scaler and transform data."""
        # Handle missing values
        if self.config.fill_missing == "forward":
            df = df.ffill()
        elif self.config.fill_missing == "backward":
            df = df.bfill()
        elif self.config.fill_missing == "interpolate":
            df = df.interpolate(method="time")
        elif self.config.fill_missing == "drop":
            df = df.dropna()

        # Separate features and targets
        if target_cols:
            target_data = df[target_cols].values
            feature_cols = [c for c in df.columns if c not in target_cols]
            feature_data = df[feature_cols].values
        else:
            feature_data = df.values
            target_data = None

        # Fit scaler on features
        self.scaler = self._get_scaler()
        feature_data_scaled = self.scaler.fit_transform(feature_data)

        self.fitted = True

        return feature_data_scaled, target_data

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.fitted or self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")

        # Handle missing values
        if self.config.fill_missing == "forward":
            df = df.ffill()
        elif self.config.fill_missing == "backward":
            df = df.bfill()
        elif self.config.fill_missing == "interpolate":
            df = df.interpolate(method="time")

        return self.scaler.transform(df.values)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self.fitted or self.scaler is None:
            raise ValueError("Scaler not fitted.")
        return self.scaler.inverse_transform(data)


def prepare_portfolio_features(
    returns: pd.DataFrame,
    lookback: int = 60,
    include_correlation: bool = True,
) -> np.ndarray:
    """Prepare features for portfolio optimization model."""
    features = []

    # Rolling statistics for each asset
    for col in returns.columns:
        asset_returns = returns[col]

        # Mean, std, skew, kurtosis over lookback
        rolling_mean = asset_returns.rolling(lookback).mean()
        rolling_std = asset_returns.rolling(lookback).std()
        rolling_skew = asset_returns.rolling(lookback).skew()
        rolling_kurt = asset_returns.rolling(lookback).kurt()

        features.extend([rolling_mean, rolling_std, rolling_skew, rolling_kurt])

    # Correlation features
    if include_correlation:
        rolling_corr = returns.rolling(lookback).corr()
        # Extract upper triangle of correlation matrix for each window
        # This is simplified - in practice, you'd want a more efficient representation
        pass

    # Stack features
    feature_array = np.column_stack([f.values for f in features])

    return feature_array


def prepare_risk_features(
    profile_data: dict,
    market_data: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Prepare features for risk assessment model."""
    features = []

    # Profile features
    features.append(profile_data.get("idade", 35) / 100)  # Normalized age
    features.append(profile_data.get("renda_mensal", 5000) / 100000)  # Normalized income
    features.append(profile_data.get("patrimonio_investido", 0) / 1000000)  # Normalized wealth
    features.append(profile_data.get("aporte_mensal", 0) / 10000)  # Normalized contribution

    # Risk profile encoding
    risk_map = {"conservador": 0, "moderado": 0.5, "arrojado": 1}
    features.append(risk_map.get(profile_data.get("perfil_risco", "moderado"), 0.5))

    # Experience encoding
    exp_map = {"iniciante": 0, "intermediario": 0.5, "avancado": 1}
    features.append(exp_map.get(profile_data.get("experiencia", "iniciante"), 0))

    # Income type encoding
    income_map = {
        "clt": 0.8, "servidor_publico": 1.0, "autonomo": 0.4,
        "empresario": 0.6, "aposentado": 0.9, "sem_renda_fixa": 0.2
    }
    features.append(income_map.get(profile_data.get("tipo_renda", "clt"), 0.5))

    # Debt flag
    features.append(float(profile_data.get("possui_dividas_caras", False)))

    # Dependents
    features.append(min(profile_data.get("dependentes", 0), 5) / 5)

    # Preferences
    features.append(float(profile_data.get("evitar_cripto", False)))
    features.append(float(profile_data.get("evitar_exterior", False)))
    features.append(float(profile_data.get("foco_renda_passiva", False)))
    features.append(float(profile_data.get("prefere_isentos_ir", False)))

    # Market context features (if available)
    if market_data is not None and not market_data.empty:
        # Add market regime indicators
        features.append(market_data.get("vix", 20).iloc[-1] / 100 if "vix" in market_data.columns else 0.2)
        features.append(market_data.get("cdi", 0.1).iloc[-1] if "cdi" in market_data.columns else 0.1)
    else:
        features.extend([0.2, 0.1])  # Defaults

    return np.array(features, dtype=np.float32).reshape(1, -1)


def create_dataloaders(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders."""
    n_samples = len(X)
    train_end = int(n_samples * train_split)
    val_end = train_end + int(n_samples * val_split)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    if y is not None:
        y_tensor = torch.FloatTensor(y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(X_tensor)

    train_dataset = torch.utils.data.Subset(dataset, range(train_end))
    val_dataset = torch.utils.data.Subset(dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(dataset, range(val_end, n_samples))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader