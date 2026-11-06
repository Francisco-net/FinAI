"""Modelos de domínio (Pydantic).

Os modelos de ENTRADA também viram o JSON Schema das ferramentas do agente, por isso usam
``Literal`` (enum inline no schema) e descrições em português nos campos.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

TipoRenda = Literal[
    "clt", "servidor_publico", "autonomo", "empresario", "aposentado", "sem_renda_fixa"
]
Experiencia = Literal["iniciante", "intermediario", "avancado"]
PerfilRisco = Literal["conservador", "moderado", "arrojado"]
Prioridade = Literal["alta", "media", "baixa"]
MetodoAlocacao = Literal["modelo", "otimizado"]
TipoBucket = Literal["reserva", "curto_prazo", "medio_prazo", "longo_prazo"]


# --------------------------------------------------------------------------- entrada
class Objetivo(BaseModel):
    nome: str = Field(
        description="Nome do objetivo, ex.: 'Entrada do apartamento', 'Aposentadoria'"
    )
    valor_alvo: float = Field(
        gt=0, description="Valor necessário em R$ de HOJE (sem considerar inflação)"
    )
    prazo_anos: float = Field(gt=0, le=60, description="Em quantos anos o dinheiro será necessário")
    prioridade: Prioridade = Field(default="media", description="Importância relativa do objetivo")


class Preferencias(BaseModel):
    evitar_cripto: bool = Field(default=False, description="Não incluir criptoativos")
    evitar_exterior: bool = Field(
        default=False, description="Não incluir investimentos no exterior"
    )
    foco_renda_passiva: bool = Field(
        default=False, description="Prefere ativos que pagam renda recorrente (FIIs, dividendos)"
    )
    prefere_isentos_ir: bool = Field(
        default=False,
        description="Prioriza produtos isentos de IR (LCI/LCA, debêntures incentivadas)",
    )


class PerfilInvestidor(BaseModel):
    """Situação financeira e preferências do usuário: entrada principal do planejador."""

    idade: int = Field(ge=16, le=100)
    renda_mensal: float = Field(ge=0, description="Renda líquida mensal em R$")
    despesas_mensais: float = Field(ge=0, description="Gasto mensal médio em R$")
    tipo_renda: TipoRenda = Field(description="Natureza da renda (define o tamanho da reserva)")
    dependentes: int = Field(default=0, ge=0, le=15)
    patrimonio_investido: float = Field(
        default=0,
        ge=0,
        description="Total disponível hoje para investir, incluindo poupança/reserva atual",
    )
    aporte_mensal: float = Field(
        default=0, ge=0, description="Quanto consegue investir por mês em R$"
    )
    possui_dividas_caras: bool = Field(
        default=False,
        description="Tem dívidas de cartão rotativo, cheque especial ou empréstimo pessoal caro",
    )
    perfil_risco: PerfilRisco = Field(
        description="Perfil de risco (idealmente vindo do questionário de suitability)"
    )
    experiencia: Experiencia = Field(default="iniciante")
    objetivos: list[Objetivo] = Field(default_factory=list)
    preferencias: Preferencias = Field(default_factory=Preferencias)
    metodo: MetodoAlocacao = Field(
        default="modelo",
        description="'modelo' = carteira-modelo ajustada (mais estável); 'otimizado' = Markowitz com volatilidade-alvo",
    )


class RespostasSuitability(BaseModel):
    """Questionário de perfil do investidor (inspirado na Resolução CVM 30/2021)."""

    objetivo: Literal[
        "preservar_capital", "renda_complementar", "crescimento_moderado", "maximizar_crescimento"
    ]
    horizonte: Literal["ate_1_ano", "1_a_3_anos", "3_a_5_anos", "5_a_10_anos", "mais_de_10_anos"]
    reacao_queda_20: Literal["vender_tudo", "vender_parte", "manter", "comprar_mais"] = Field(
        description="O que faria se a carteira caísse 20% em poucos meses"
    )
    conhecimento: Literal["nenhum", "basico", "intermediario", "avancado"] = Field(
        description="nenhum; basico (poupança/CDB); intermediario (Tesouro, fundos, ETFs); avancado (ações, derivativos)"
    )
    experiencia_anos: Literal["nunca", "menos_de_2", "2_a_5", "mais_de_5"]
    estabilidade_renda: Literal["instavel", "variavel", "estavel", "muito_estavel"]
    necessidade_liquidez: Literal["alta", "media", "baixa", "nenhuma"] = Field(
        description="Chance de precisar resgatar boa parte do dinheiro nos próximos 12 meses"
    )


# --------------------------------------------------------------------------- saída
class ResultadoSuitability(BaseModel):
    perfil: PerfilRisco
    pontuacao: float
    perfil_pela_pontuacao: PerfilRisco
    travas_aplicadas: list[str]
    experiencia_sugerida: Experiencia
    descricao: str


class ReservaEmergencia(BaseModel):
    meses_recomendados: int
    valor_alvo: float
    valor_alocado: float
    falta: float
    meses_para_completar: float | None
    onde_investir: list[str]


class Bucket(BaseModel):
    """Uma "caixinha" da carteira, com horizonte e alocação próprios."""

    nome: str
    tipo: TipoBucket
    prazo_anos: float | None = None
    valor_alvo: float | None = None
    valor_alocado: float = 0
    aporte_mensal: float = 0
    aporte_necessario: float | None = None
    alocacao: dict[str, float]  # classe -> % (0-100)
    status: str = ""


class LinhaAlocacao(BaseModel):
    classe: str
    nome: str
    percentual: float
    valor: float
    veiculos: list[str]


class Metricas(BaseModel):
    retorno_esperado_nominal: float
    retorno_esperado_real: float
    volatilidade: float
    sharpe: float | None
    perda_1_ano_cenario_ruim: float = Field(description="VaR 95% paramétrico em 1 ano (fração)")
    prob_retorno_negativo_1_ano: float


class AporteDestino(BaseModel):
    destino: str
    valor_mensal: float
    fase: Literal["1_reserva", "2_objetivos"]


class Carteira(BaseModel):
    perfil_risco: PerfilRisco
    metodo: MetodoAlocacao
    patrimonio_total: float
    aporte_mensal_total: float
    reserva: ReservaEmergencia
    buckets: list[Bucket]
    alocacao_consolidada: list[LinhaAlocacao]
    metricas: Metricas
    plano_aportes: list[AporteDestino]
    alertas: list[str]
    proximos_passos: list[str]
    premissas: dict[str, str | float]


class SimulacaoEntrada(BaseModel):
    valor_inicial: float = Field(ge=0, description="Patrimônio inicial em R$")
    aporte_mensal: float = Field(
        ge=0, description="Aporte mensal em R$ de hoje (reajustado pela inflação)"
    )
    anos: float = Field(gt=0, le=60)
    perfil_risco: PerfilRisco | None = Field(
        default=None, description="Usa a carteira-modelo do perfil se 'alocacao' não for informada"
    )
    alocacao: dict[str, float] | None = Field(
        default=None,
        description="Alocação explícita: classe -> % (ex.: {'rf_pos': 60, 'acoes_br': 40})",
    )
    meta: float | None = Field(
        default=None, gt=0, description="Valor desejado ao final, em R$ de hoje"
    )

    @model_validator(mode="after")
    def _check_source(self) -> SimulacaoEntrada:
        if self.perfil_risco is None and not self.alocacao:
            raise ValueError("Informe 'perfil_risco' ou 'alocacao'")
        return self
