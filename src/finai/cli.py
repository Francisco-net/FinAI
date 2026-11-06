"""Linha de comando do FinAI.

Exemplos:
    finai chat                                   # conversa com o assistente (requer ANTHROPIC_API_KEY)
    finai perfil                                 # questionário de suitability (offline)
    finai carteira examples/perfil_exemplo.json  # carteira sem IA, a partir de um JSON
    finai simular --inicial 10000 --aporte 1000 --anos 20 --perfil moderado --meta 1000000
    finai indicadores
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, get_args

import click
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from finai.domain.models import Carteira, PerfilInvestidor, PerfilRisco, RespostasSuitability
from finai.domain.suitability import avaliar_perfil
from finai.market.bcb import get_indicators
from finai.portfolio.allocation import model_portfolio
from finai.portfolio.metrics import portfolio_return, portfolio_volatility
from finai.portfolio.planner import montar_carteira
from finai.portfolio.simulation import simulate

app = typer.Typer(help="FinAI - assistente de investimentos com IA", no_args_is_help=True)
console = Console()


SIGLAS = {
    "rf_pos": "Pós",
    "rf_pre": "Pré",
    "rf_inflacao": "IPCA+",
    "rf_credito": "Crédito",
    "multimercado": "Multi",
    "acoes_br": "Ações BR",
    "fii": "FII",
    "internacional": "Exterior",
    "cripto": "Cripto",
}


def resumo_alocacao(alocacao: dict[str, float]) -> str:
    itens = sorted(alocacao.items(), key=lambda kv: -kv[1])
    return " · ".join(f"{SIGLAS.get(k, k)} {v:.0f}%" for k, v in itens)


def brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    return f"{v * 100:.1f}%".replace(".", ",")


# ---------------------------------------------------------------------------- chat
@app.command()
def chat(
    mostrar_raciocinio: Annotated[
        bool, typer.Option(help="Exibe o resumo do raciocínio do modelo")
    ] = False,
) -> None:
    """Conversa com o assistente de IA (digite 'sair' para encerrar, 'nova' para recomeçar)."""
    from finai.agent.engine import FinAIAgent

    try:
        agent = FinAIAgent()
    except Exception as exc:  # credenciais ausentes etc.
        console.print(f"[red]Não foi possível iniciar o cliente da Anthropic:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        Panel.fit(
            "[bold]FinAI[/] - seu assistente de investimentos\n"
            "Conte sua situação e seus objetivos. Comandos: [cyan]nova[/] (recomeçar), [cyan]sair[/].",
            border_style="green",
        )
    )
    while True:
        try:
            user = console.input("\n[bold green]Você ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in {"sair", "exit", "quit"}:
            break
        if user.lower() == "nova":
            agent.reset()
            console.print("[dim]Conversa reiniciada.[/]")
            continue

        console.print("\n[bold blue]FinAI ›[/]")
        buffer = ""
        with Live(
            Markdown(""), console=console, refresh_per_second=12, vertical_overflow="visible"
        ) as live:
            for ev in agent.chat(user):
                if ev.kind == "text":
                    buffer += ev.text
                    live.update(Markdown(buffer))
                elif ev.kind == "thinking" and mostrar_raciocinio:
                    console.print(f"[dim italic]{ev.text}[/]", end="")
                elif ev.kind == "tool_start":
                    console.print(f"[dim]› consultando {ev.tool_name}...[/]")
                elif ev.kind == "tool_result" and ev.is_error:
                    console.print(f"[yellow]› {ev.tool_name}: {ev.data}[/]")
                elif ev.kind == "refusal":
                    buffer = ev.text
                    live.update(Markdown(buffer))
                elif ev.kind in {"notice", "error"}:
                    console.print(f"[{'yellow' if ev.kind == 'notice' else 'red'}]{ev.text}[/]")
        u = agent.usage
        console.print(
            f"[dim]tokens: entrada {u.input_tokens} · cache {u.cache_read_input_tokens} · saída {u.output_tokens}[/]"
        )


# ---------------------------------------------------------------------------- perfil
def _choose(question: str, options: tuple[str, ...]) -> str:
    console.print(f"\n[bold]{question}[/]")
    for i, opt in enumerate(options, 1):
        console.print(f"  {i}. {opt.replace('_', ' ')}")
    idx = typer.prompt("Escolha", type=click.IntRange(1, len(options)))
    return options[idx - 1]


@app.command()
def perfil() -> None:
    """Questionário de perfil de investidor (suitability), sem IA."""
    fields = RespostasSuitability.model_fields
    perguntas = {
        "objetivo": "Qual seu principal objetivo ao investir?",
        "horizonte": "Por quanto tempo pretende manter o dinheiro investido?",
        "reacao_queda_20": "Se sua carteira caísse 20% em poucos meses, você...",
        "conhecimento": "Qual seu conhecimento sobre investimentos?",
        "experiencia_anos": "Há quanto tempo você investe?",
        "estabilidade_renda": "Como é a estabilidade da sua renda?",
        "necessidade_liquidez": "Qual a chance de precisar de boa parte do dinheiro em 12 meses?",
    }
    respostas = {
        campo: _choose(texto, get_args(fields[campo].annotation))
        for campo, texto in perguntas.items()
    }
    res = avaliar_perfil(RespostasSuitability(**respostas))
    console.print(
        Panel(
            f"[bold]Perfil: {res.perfil.upper()}[/] (pontuação {res.pontuacao}/100)\n\n{res.descricao}"
            + (
                "\n\n[yellow]Travas:[/] " + "; ".join(res.travas_aplicadas)
                if res.travas_aplicadas
                else ""
            )
            + f"\n\nExperiência sugerida: {res.experiencia_sugerida}",
            title="Resultado",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------- carteira
def print_carteira(c: Carteira) -> None:
    console.print(
        Panel.fit(
            f"Perfil [bold]{c.perfil_risco}[/] · método {c.metodo} · patrimônio {brl(c.patrimonio_total)} · "
            f"aporte {brl(c.aporte_mensal_total)}/mês",
            title="Carteira FinAI",
            border_style="green",
        )
    )

    t = Table(title="Caixinhas (objetivos)")
    for col in ("Caixinha", "Prazo", "Meta", "Alocado hoje", "Aporte/mês", "Alocação", "Status"):
        t.add_column(col)
    for b in c.buckets:
        t.add_row(
            b.nome,
            f"{b.prazo_anos:g} anos" if b.prazo_anos else "-",
            brl(b.valor_alvo) if b.valor_alvo else "-",
            brl(b.valor_alocado),
            brl(b.aporte_mensal),
            resumo_alocacao(b.alocacao),
            b.status,
        )
    console.print(t)

    t = Table(title="Alocação consolidada")
    for col in ("Classe", "%", "Valor", "Exemplos"):
        t.add_column(col)
    for linha in c.alocacao_consolidada:
        t.add_row(linha.nome, pct(linha.percentual / 100), brl(linha.valor), linha.veiculos[0])
    console.print(t)

    m = c.metricas
    console.print(
        f"Retorno esperado: {pct(m.retorno_esperado_nominal)} a.a. ({pct(m.retorno_esperado_real)} real) · "
        f"Volatilidade: {pct(m.volatilidade)} · Perda em ano ruim (VaR 95%): {pct(m.perda_1_ano_cenario_ruim)}"
    )
    if c.alertas:
        console.print(
            Panel("\n".join(f"• {a}" for a in c.alertas), title="Alertas", border_style="yellow")
        )
    console.print(
        Panel(
            "\n".join(f"{i}. {p}" for i, p in enumerate(c.proximos_passos, 1)),
            title="Próximos passos",
            border_style="blue",
        )
    )


@app.command()
def carteira(
    arquivo: Annotated[
        Path, typer.Argument(help="JSON com o perfil (veja examples/perfil_exemplo.json)")
    ],
    metodo: Annotated[str | None, typer.Option(help="modelo | otimizado")] = None,
    json_saida: Annotated[bool, typer.Option("--json", help="Imprime o resultado em JSON")] = False,
) -> None:
    """Monta a carteira a partir de um perfil em JSON (sem IA)."""
    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    if metodo:
        dados["metodo"] = metodo
    c = montar_carteira(PerfilInvestidor.model_validate(dados))
    if json_saida:
        console.print_json(c.model_dump_json())
    else:
        print_carteira(c)


# ---------------------------------------------------------------------------- simulação
@app.command()
def simular(
    inicial: Annotated[float, typer.Option(help="Valor inicial (R$)")] = 0.0,
    aporte: Annotated[float, typer.Option(help="Aporte mensal (R$ de hoje)")] = 1000.0,
    anos: Annotated[float, typer.Option(help="Prazo em anos")] = 20.0,
    perfil: Annotated[str, typer.Option(help="conservador | moderado | arrojado")] = "moderado",
    meta: Annotated[float | None, typer.Option(help="Meta em R$ de hoje")] = None,
) -> None:
    """Projeção de patrimônio (Monte Carlo, em R$ de hoje)."""
    if perfil not in get_args(PerfilRisco):
        raise typer.BadParameter("perfil deve ser conservador, moderado ou arrojado")
    perfil_risco: Literal["conservador", "moderado", "arrojado"] = perfil  # type: ignore[assignment]
    w = model_portfolio(perfil_risco)
    r = simulate(inicial, aporte, anos, portfolio_return(w), portfolio_volatility(w), meta)
    t = Table(
        title=f"Projeção - perfil {perfil} ({pct(r.retorno_real_esperado)} real a.a. esperado)"
    )
    for col in ("Ano", "Pessimista (p10)", "Mediano (p50)", "Otimista (p90)", "Total aportado"):
        t.add_column(col, justify="right")
    passo = max(1, len(r.serie_anual) // 10)
    for p in r.serie_anual[::passo] + (
        [r.serie_anual[-1]] if (len(r.serie_anual) - 1) % passo else []
    ):
        t.add_row(str(p.ano), brl(p.p10), brl(p.p50), brl(p.p90), brl(p.total_aportado))
    console.print(t)
    if meta:
        console.print(
            f"Probabilidade de atingir {brl(meta)}: [bold]{pct(r.prob_atingir_meta or 0)}[/] · "
            f"aporte necessário (cenário esperado): {brl(r.aporte_necessario_para_meta or 0)}/mês"
        )
    console.print(f"[dim]{r.observacao}[/]")


# ---------------------------------------------------------------------------- indicadores
@app.command()
def indicadores() -> None:
    """Selic, CDI, IPCA e dólar (Banco Central)."""
    ind = get_indicators()
    t = Table(title="Indicadores")
    for col in ("Indicador", "Valor", "Data", "Fonte"):
        t.add_column(col)
    t.add_row("Selic meta", pct(ind.selic_meta.valor), ind.selic_meta.data, ind.selic_meta.fonte)
    t.add_row("CDI", pct(ind.cdi_anual.valor), ind.cdi_anual.data, ind.cdi_anual.fonte)
    t.add_row("IPCA 12m", pct(ind.ipca_12m.valor), ind.ipca_12m.data, ind.ipca_12m.fonte)
    t.add_row("Dólar PTAX", brl(ind.dolar_ptax.valor), ind.dolar_ptax.data, ind.dolar_ptax.fonte)
    t.add_row("Juro real (ex-post)", pct(ind.juro_real_ex_post), "", "")
    console.print(t)
    if ind.aviso:
        console.print(f"[yellow]{ind.aviso}[/]")


if __name__ == "__main__":
    app()
