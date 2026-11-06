"""Database connection and session management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from finai.config import Settings


class DatabaseManager:
    """Gerencia conexões com o banco de dados."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            # Use NullPool for serverless environments, otherwise use default pool
            pool_class = NullPool if self.settings.database_pool_disabled else None
            self._engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.database_echo,
                poolclass=pool_class,
                pool_pre_ping=True,
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Fornece uma sessão de banco de dados."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Fecha o engine do banco de dados."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def create_tables(self) -> None:
        """Cria todas as tabelas (apenas para desenvolvimento/testes)."""
        from finai.database.models import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Remove todas as tabelas (apenas para testes)."""
        from finai.database.models import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# Instância global (será inicializada na aplicação)
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Retorna o gerenciador de banco de dados global."""
    global _db_manager
    if _db_manager is None:
        from finai.config import get_settings
        _db_manager = DatabaseManager(get_settings())
    return _db_manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency para FastAPI - fornece uma sessão de banco de dados."""
    db_manager = get_db_manager()
    async with db_manager.session() as session:
        yield session