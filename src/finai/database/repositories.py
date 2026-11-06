"""Repository layer for database operations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from finai.database.models import (
    User,
    InvestorProfile,
    FinancialObjective,
    Portfolio,
    Simulation,
    ChatSession,
    ChatMessage,
    MarketData,
    EconomicIndicator,
    FinancialKnowledge,
    UserRole,
    SimulationStatus,
    ChatMessageRole,
)


class UserRepository:
    """Repositório de usuários."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        role: UserRole = UserRole.USER,
    ) -> User:
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def update(self, user: User) -> User:
        user.updated_at = datetime.utcnow()
        await self.session.flush()
        return user

    async def delete(self, user_id: str) -> bool:
        user = await self.get_by_id(user_id)
        if user:
            await self.session.delete(user)
            return True
        return False


class InvestorProfileRepository:
    """Repositório de perfis de investidor."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, profile: InvestorProfile) -> InvestorProfile:
        self.session.add(profile)
        await self.session.flush()
        return profile

    async def get_by_id(self, profile_id: str) -> Optional[InvestorProfile]:
        result = await self.session.execute(
            select(InvestorProfile).where(InvestorProfile.id == profile_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(self, user_id: str) -> list[InvestorProfile]:
        result = await self.session.execute(
            select(InvestorProfile)
            .where(InvestorProfile.user_id == user_id)
            .order_by(desc(InvestorProfile.created_at))
        )
        return list(result.scalars().all())

    async def update(self, profile: InvestorProfile) -> InvestorProfile:
        profile.updated_at = datetime.utcnow()
        await self.session.flush()
        return profile

    async def delete(self, profile_id: str) -> bool:
        profile = await self.get_by_id(profile_id)
        if profile:
            await self.session.delete(profile)
            return True
        return False


class PortfolioRepository:
    """Repositório de carteiras."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, portfolio: Portfolio) -> Portfolio:
        self.session.add(portfolio)
        await self.session.flush()
        return portfolio

    async def get_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        result = await self.session.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[Portfolio]:
        result = await self.session.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user_id)
            .order_by(desc(Portfolio.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_latest_by_user(self, user_id: str) -> Optional[Portfolio]:
        result = await self.session.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user_id)
            .order_by(desc(Portfolio.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def count_by_user(self, user_id: str) -> int:
        result = await self.session.execute(
            select(func.count(Portfolio.id)).where(Portfolio.user_id == user_id)
        )
        return result.scalar_one()

    async def update(self, portfolio: Portfolio) -> Portfolio:
        portfolio.updated_at = datetime.utcnow()
        await self.session.flush()
        return portfolio

    async def delete(self, portfolio_id: str) -> bool:
        portfolio = await self.get_by_id(portfolio_id)
        if portfolio:
            await self.session.delete(portfolio)
            return True
        return False


class SimulationRepository:
    """Repositório de simulações."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, simulation: Simulation) -> Simulation:
        self.session.add(simulation)
        await self.session.flush()
        return simulation

    async def get_by_id(self, simulation_id: str) -> Optional[Simulation]:
        result = await self.session.execute(
            select(Simulation).where(Simulation.id == simulation_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[Simulation]:
        result = await self.session.execute(
            select(Simulation)
            .where(Simulation.user_id == user_id)
            .order_by(desc(Simulation.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_by_portfolio(self, portfolio_id: str) -> list[Simulation]:
        result = await self.session.execute(
            select(Simulation)
            .where(Simulation.portfolio_id == portfolio_id)
            .order_by(desc(Simulation.created_at))
        )
        return list(result.scalars().all())

    async def update(self, simulation: Simulation) -> Simulation:
        if simulation.status == SimulationStatus.COMPLETED and simulation.completed_at is None:
            simulation.completed_at = datetime.utcnow()
        await self.session.flush()
        return simulation

    async def delete(self, simulation_id: str) -> bool:
        simulation = await self.get_by_id(simulation_id)
        if simulation:
            await self.session.delete(simulation)
            return True
        return False


class ChatSessionRepository:
    """Repositório de sessões de chat."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, chat_session: ChatSession) -> ChatSession:
        self.session.add(chat_session)
        await self.session.flush()
        return chat_session

    async def get_by_id(self, session_id: str) -> Optional[ChatSession]:
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[ChatSession]:
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.updated_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_active_by_user(self, user_id: str) -> Optional[ChatSession]:
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.is_active == True)
            .order_by(desc(ChatSession.updated_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update(self, chat_session: ChatSession) -> ChatSession:
        chat_session.updated_at = datetime.utcnow()
        await self.session.flush()
        return chat_session

    async def delete(self, session_id: str) -> bool:
        session = await self.get_by_id(session_id)
        if session:
            await self.session.delete(session)
            return True
        return False


class ChatMessageRepository:
    """Repositório de mensagens de chat."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        session_id: str,
        role: ChatMessageRole,
        content: str,
        thinking: Optional[str] = None,
        tool_calls: Optional[list] = None,
        tool_results: Optional[list] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            tool_results=tool_results,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def get_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[ChatMessage]:
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_recent_by_session(
        self, session_id: str, limit: int = 20
    ) -> list[ChatMessage]:
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(limit)
        )
        messages = list(result.scalars().all())
        return list(reversed(messages))  # Ordem cronológica


class MarketDataRepository:
    """Repositório de dados de mercado."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, market_data: MarketData) -> MarketData:
        self.session.add(market_data)
        await self.session.flush()
        return market_data

    async def get_latest_by_symbol(self, symbol: str) -> Optional[MarketData]:
        result = await self.session.execute(
            select(MarketData)
            .where(MarketData.symbol == symbol)
            .order_by(desc(MarketData.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_history(
        self, symbol: str, days: int = 30
    ) -> list[MarketData]:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(MarketData)
            .where(MarketData.symbol == symbol, MarketData.timestamp >= cutoff)
            .order_by(MarketData.timestamp)
        )
        return list(result.scalars().all())


class EconomicIndicatorRepository:
    """Repositório de indicadores econômicos."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, indicator: EconomicIndicator) -> EconomicIndicator:
        self.session.add(indicator)
        await self.session.flush()
        return indicator

    async def get_latest_by_code(self, code: str) -> Optional[EconomicIndicator]:
        result = await self.session.execute(
            select(EconomicIndicator)
            .where(EconomicIndicator.code == code)
            .order_by(desc(EconomicIndicator.date))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_history(
        self, code: str, limit: int = 100
    ) -> list[EconomicIndicator]:
        result = await self.session.execute(
            select(EconomicIndicator)
            .where(EconomicIndicator.code == code)
            .order_by(desc(EconomicIndicator.date))
            .limit(limit)
        )
        return list(result.scalars().all())


class FinancialKnowledgeRepository:
    """Repositório de conhecimento financeiro."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, knowledge: FinancialKnowledge) -> FinancialKnowledge:
        self.session.add(knowledge)
        await self.session.flush()
        return knowledge

    async def get_by_term(self, term: str) -> Optional[FinancialKnowledge]:
        result = await self.session.execute(
            select(FinancialKnowledge).where(FinancialKnowledge.term == term)
        )
        return result.scalar_one_or_none()

    async def search(self, query: str, limit: int = 10) -> list[FinancialKnowledge]:
        """Busca por termo (pode ser estendido para busca vetorial)."""
        result = await self.session.execute(
            select(FinancialKnowledge)
            .where(
                FinancialKnowledge.term.ilike(f"%{query}%") |
                FinancialKnowledge.definition.ilike(f"%{query}%")
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_category(self, category: str) -> list[FinancialKnowledge]:
        result = await self.session.execute(
            select(FinancialKnowledge)
            .where(FinancialKnowledge.category == category)
            .order_by(FinancialKnowledge.term)
        )
        return list(result.scalars().all())