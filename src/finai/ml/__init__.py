"""Machine Learning module for FinAI."""

from finai.ml.models import (
    TimeSeriesTransformer,
    LSTMForecaster,
    GRUForecaster,
    MLPPortfolioOptimizer,
    RiskAssessmentNet,
    ReturnPredictor,
    MarketRegimeDetector,
    EnsembleModel,
    ModelConfig,
    create_return_predictor,
    create_portfolio_optimizer,
    create_risk_assessment,
    create_market_regime_detector,
)
from finai.ml.training import (
    TrainingConfig,
    TrainingHistory,
    EarlyStopping,
    get_device,
    create_optimizer,
    create_scheduler,
    train_epoch,
    validate_epoch,
    train_model,
    predict,
    save_model,
    load_model,
)
from finai.ml.data import (
    FeatureConfig,
    FinancialFeatureEngineer,
    prepare_portfolio_features,
    prepare_risk_features,
    create_dataloaders,
)
from finai.ml.inference import (
    ModelArtifacts,
    ModelRegistry,
    InferenceEngine,
    create_default_models,
    initialize_inference_engine,
)

__all__ = [
    # Models
    "TimeSeriesTransformer",
    "LSTMForecaster",
    "GRUForecaster",
    "MLPPortfolioOptimizer",
    "RiskAssessmentNet",
    "ReturnPredictor",
    "MarketRegimeDetector",
    "EnsembleModel",
    "ModelConfig",
    "create_return_predictor",
    "create_portfolio_optimizer",
    "create_risk_assessment",
    "create_market_regime_detector",
    # Training
    "TrainingConfig",
    "TrainingHistory",
    "EarlyStopping",
    "get_device",
    "create_optimizer",
    "create_scheduler",
    "train_epoch",
    "validate_epoch",
    "train_model",
    "predict",
    "save_model",
    "load_model",
    # Data
    "FeatureConfig",
    "FinancialFeatureEngineer",
    "prepare_portfolio_features",
    "prepare_risk_features",
    "create_dataloaders",
    # Inference
    "ModelArtifacts",
    "ModelRegistry",
    "InferenceEngine",
    "create_default_models",
    "initialize_inference_engine",
]