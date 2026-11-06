"""Base de conhecimento: premissas de mercado, classes de ativos, tributação e glossário."""

from finai.knowledge.assets import ASSET_CLASSES, AssetClassInfo, load_assumptions

__all__ = ["ASSET_CLASSES", "AssetClassInfo", "load_assumptions"]
