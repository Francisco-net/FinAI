"""Database models using SQLAlchemy."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class RiskProfile(str, enum.Enum):
    CONSERVADOR = "conservador"
    MODERADO = "moderado"
    ARROJADO = "arrojado"


class IncomeType(str, enum.Enum):
    CLT = "clt"
    SERVIDOR_PUBLICO = "servidor_publico"
    AUTONOMO = "autonomo"
    EMPRESARIO = "empresario"
    APOSENTADO = "aposentado"
    SEM_RENDA_FIXA = "sem_renda_fixa"


class ExperienceLevel(str, enum.Enum):
    INICIANTE = "iniciante"
    INTERMEDIARIO = "intermediario"
    AVANCADO = "avancado"


class PriorityLevel(str, enum.Enum):
    ALTA = "alta"
    MEDIA = "media"
    BAIXA = "baixa"


class AllocationMethod(str, enum.Enum):
    MODELO = "modelo"
    OTIMIZADO = "otimizado"


class BucketType(str, enum.Enum):
    RESERVA = "reserva"
    CURTO_PRAZO = "curto_prazo"
    MEDIO_PRAZO = "medio_prazo"
    LONGO_PRAZO = "longo_prazo"


class SimulationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatMessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    profiles: Mapped[list["InvestorProfile"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    simulations: Mapped[list["Simulation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class InvestorProfile(Base):
    __tablename__ = "investor_profiles"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Dados pessoais
    idade: Mapped[int] = mapped_column(Integer, nullable=False)
    renda_mensal: Mapped[float] = mapped_column(Float, nullable=False)
    despesas_mensais: Mapped[float] = mapped_column(Float, nullable=False)
    tipo_renda: Mapped[IncomeType] = mapped_column(Enum(IncomeType), nullable=False)
    dependentes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Patrimônio e aporte
    patrimonio_investido: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    aporte_mensal: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    possui_dividas_caras: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Perfil de risco
    perfil_risco: Mapped[RiskProfile] = mapped_column(Enum(RiskProfile), nullable=False)
    experiencia: Mapped[ExperienceLevel] = mapped_column(Enum(ExperienceLevel), default=ExperienceLevel.INICIANTE, nullable=False)

    # Preferências
    evitar_cripto: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    evitar_exterior: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    foco_renda_passiva: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    prefere_isentos_ir: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Método de alocação
    metodo: Mapped[AllocationMethod] = mapped_column(Enum(AllocationMethod), default=AllocationMethod.MODELO, nullable=False)

    # Metadados
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="profiles")
    objectives: Mapped[list["FinancialObjective"]] = relationship(back_populates="profile", cascade="all, delete-orphan")


class FinancialObjective(Base):
    __tablename__ = "financial_objectives"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    profile_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("investor_profiles.id"), nullable=False, index=True)
    nome: Mapped[str] = mapped_column(String(200), nullable=False)
    valor_alvo: Mapped[float] = mapped_column(Float, nullable=False)
    prazo_anos: Mapped[float] = mapped_column(Float, nullable=False)
    prioridade: Mapped[PriorityLevel] = mapped_column(Enum(PriorityLevel), default=PriorityLevel.MEDIA, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    profile: Mapped["InvestorProfile"] = relationship(back_populates="objectives")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    profile_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("investor_profiles.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Dados da carteira
    perfil_risco: Mapped[RiskProfile] = mapped_column(Enum(RiskProfile), nullable=False)
    metodo: Mapped[AllocationMethod] = mapped_column(Enum(AllocationMethod), nullable=False)
    patrimonio_total: Mapped[float] = mapped_column(Float, nullable=False)
    aporte_mensal_total: Mapped[float] = mapped_column(Float, nullable=False)

    # JSON com dados completos
    reserva_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    buckets_data: Mapped[list] = mapped_column(JSON, nullable=False)
    alocacao_consolidada: Mapped[list] = mapped_column(JSON, nullable=False)
    metricas: Mapped[dict] = mapped_column(JSON, nullable=False)
    plano_aportes: Mapped[list] = mapped_column(JSON, nullable=False)
    alertas: Mapped[list] = mapped_column(JSON, nullable=False)
    proximos_passos: Mapped[list] = mapped_column(JSON, nullable=False)
    premissas: Mapped[dict] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="portfolios")
    profile: Mapped["InvestorProfile"] = relationship()


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    portfolio_id: Mapped[Optional[str]] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("portfolios.id"), nullable=True, index=True)

    # Parâmetros de entrada
    valor_inicial: Mapped[float] = mapped_column(Float, nullable=False)
    aporte_mensal: Mapped[float] = mapped_column(Float, nullable=False)
    anos: Mapped[float] = mapped_column(Float, nullable=False)
    perfil_risco: Mapped[Optional[RiskProfile]] = mapped_column(Enum(RiskProfile), nullable=True)
    alocacao: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    meta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Resultados
    status: Mapped[SimulationStatus] = mapped_column(Enum(SimulationStatus), default=SimulationStatus.PENDING, nullable=False)
    serie_anual: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    p10_final: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p50_final: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p90_final: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_atingir_meta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    aporte_necessario: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_aportado: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="simulations")
    portfolio: Mapped[Optional["Portfolio"]] = relationship()


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Última carteira montada na sessão
    last_portfolio: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    last_simulation: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role: Mapped[ChatMessageRole] = mapped_column(Enum(ChatMessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadados do agente
    thinking: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tokens_input: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class MarketData(Base):
    __tablename__ = "market_data"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    symbol: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    change_percent: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(10), default="BRL", nullable=False)

    # Dados adicionais
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_market_data_symbol_timestamp", "symbol", "timestamp"),
        UniqueConstraint("symbol", "timestamp", name="uq_market_data_symbol_timestamp"),
    )


class EconomicIndicator(Base):
    __tablename__ = "economic_indicators"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    frequency: Mapped[str] = mapped_column(String(20), nullable=False)  # daily, monthly, yearly

    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_economic_indicators_code_date", "code", "date"),
        UniqueConstraint("code", "date", name="uq_economic_indicator_code_date"),
    )


class FinancialKnowledge(Base):
    __tablename__ = "financial_knowledge"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    term: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    synonyms: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    related_terms: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # For vector search

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_financial_knowledge_category", "category"),
    )