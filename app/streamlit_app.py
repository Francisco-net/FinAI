"""Interface web do FinAI (Streamlit).

Rodar: uv run streamlit run app/streamlit_app.py

Abas:
  - Assistente: chat com o agente (requer ANTHROPIC_API_KEY) + painel da carteira montada.
  - Planejador sem IA: formulário que usa o mesmo motor de carteiras, offline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finai.domain.models import Objetivo, PerfilInvestidor, Preferencias
from finai.portfolio.allocation import model_portfolio
from finai.portfolio.metrics import portfolio_return, portfolio_volatility
from finai.portfolio.planner import montar_carteira
from finai.portfolio.simulation import simulate

st.set_page_config(page_title="FinAI - Assistente de Investimentos", page_icon="📈", layout="wide")


def pct(v: float) -> str:
    return f"{v * 100:.1f}%".replace(".", ",")


def brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------- visualizações
def render_carteira(c: dict[str, Any]) -> None:
    m = c["metricas"]
    cols = st.columns(4)
    cols[0].metric("Patrimônio", brl(c["patrimonio_total"]))
    cols[1].metric("Aporte mensal", brl(c["aporte_mensal_total"]))
    cols[2].metric("Retorno esperado (real)", f"{pct(m['retorno_esperado_real'])} a.a.")
    cols[3].metric("Volatilidade", pct(m["volatilidade"]))

    df = pd.DataFrame(c["alocacao_consolidada"])
    if not df.empty:
        left, right = st.columns([1, 1])
        fig = px.pie(df, names="nome", values="percentual", hole=0.5, title="Alocação consolidada")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin={"t": 40, "b": 0, "l": 0, "r": 0})
        left.plotly_chart(fig, use_container_width=True)
        right.dataframe(
            df.assign(exemplo=df["veiculos"].str[0])[
                ["nome", "percentual", "valor", "exemplo"]
            ].rename(
                columns={
                    "nome": "Classe",
                    "percentual": "%",
                    "valor": "Valor (R$)",
                    "exemplo": "Exemplo de produto",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Caixinhas por objetivo")
    st.dataframe(
        pd.DataFrame(c["buckets"])[
            ["nome", "prazo_anos", "valor_alvo", "valor_alocado", "aporte_mensal", "status"]
        ].rename(
            columns={
                "nome": "Caixinha",
                "prazo_anos": "Prazo (anos)",
                "valor_alvo": "Meta",
                "valor_alocado": "Alocado hoje",
                "aporte_mensal": "Aporte/mês",
                "status": "Status",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    if c["alertas"]:
        with st.expander("⚠️ Alertas", expanded=True):
            for a in c["alertas"]:
                st.markdown(f"- {a}")
    with st.expander("✅ Próximos passos"):
        for i, p in enumerate(c["proximos_passos"], 1):
            st.markdown(f"{i}. {p}")


def render_simulacao(s: dict[str, Any], meta: float | None = None) -> None:
    df = pd.DataFrame(s["serie_anual"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ano"], y=df["p90"], name="Otimista (p90)", line={"width": 0}, showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ano"], y=df["p10"], name="Faixa p10-p90", fill="tonexty", line={"width": 0}
        )
    )
    fig.add_trace(go.Scatter(x=df["ano"], y=df["p50"], name="Mediano (p50)", line={"width": 3}))
    fig.add_trace(
        go.Scatter(x=df["ano"], y=df["total_aportado"], name="Total aportado", line={"dash": "dot"})
    )
    if meta:
        fig.add_hline(y=meta, line_dash="dash", annotation_text="Meta")
    fig.update_layout(
        title="Projeção do patrimônio (R$ de hoje)",
        xaxis_title="Ano",
        yaxis_title="R$",
        hovermode="x unified",
        margin={"t": 40},
    )
    st.plotly_chart(fig, use_container_width=True)
    if s.get("prob_atingir_meta") is not None:
        st.info(f"Probabilidade de atingir a meta: **{s['prob_atingir_meta'] * 100:.0f}%**")


# ---------------------------------------------------------------------------- aba: assistente
def tab_assistente() -> None:
    from finai.agent.engine import FinAIAgent

    if "agent" not in st.session_state:
        try:
            st.session_state.agent = FinAIAgent()
        except Exception as exc:
            st.error(
                f"Não foi possível iniciar o agente: {exc}. Configure ANTHROPIC_API_KEY no .env."
            )
            return
        st.session_state.chat = []  # [(papel, texto)] apenas para exibição
    agent: FinAIAgent = st.session_state.agent

    with st.sidebar:
        st.header("Sessão")
        if st.button("🔄 Nova conversa", use_container_width=True):
            agent.reset()
            st.session_state.chat = []
            st.rerun()
        u = agent.usage
        st.caption(
            f"Tokens: entrada {u.input_tokens:,} · cache {u.cache_read_input_tokens:,} · saída {u.output_tokens:,}"
        )

    chat_col, painel_col = st.columns([3, 2]) if agent.last_portfolio else (st.container(), None)

    with chat_col:
        if not st.session_state.chat:
            st.markdown(
                "👋 Olá! Sou o **FinAI**. Me conte sobre você: idade, renda, quanto já investe e quais são "
                'seus objetivos (ex.: *"Tenho 28 anos, ganho R$ 6 mil, gasto R$ 4 mil e quero comprar um '
                'apartamento em 5 anos"*).'
            )
        for papel, texto in st.session_state.chat:
            st.chat_message(papel).markdown(texto)

    if prompt := st.chat_input("Escreva sua mensagem..."):
        st.session_state.chat.append(("user", prompt))
        with chat_col:
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                status = st.status("Pensando...", expanded=False)
                placeholder = st.empty()
                buffer = ""
                for ev in agent.chat(prompt):
                    if ev.kind == "text":
                        buffer += ev.text
                        placeholder.markdown(buffer + "▌")
                    elif ev.kind == "thinking":
                        status.write(ev.text)
                    elif ev.kind == "tool_start":
                        status.update(label=f"Consultando {ev.tool_name}...")
                    elif ev.kind == "tool_result" and ev.is_error:
                        status.write(f"⚠️ {ev.tool_name}: {ev.data}")
                    elif ev.kind == "refusal":
                        buffer = ev.text
                    elif ev.kind == "error":
                        st.error(ev.text)
                    elif ev.kind == "notice":
                        st.warning(ev.text)
                status.update(label="Concluído", state="complete")
                placeholder.markdown(buffer)
        st.session_state.chat.append(("assistant", buffer))
        st.rerun()

    if painel_col is not None and agent.last_portfolio:
        with painel_col:
            st.subheader("📊 Sua carteira")
            render_carteira(agent.last_portfolio)
            if agent.last_simulation:
                render_simulacao(agent.last_simulation)


# ---------------------------------------------------------------------------- aba: planejador offline
def tab_planejador() -> None:
    st.caption("Monte uma carteira com o mesmo motor do assistente, sem usar IA.")
    with st.form("perfil"):
        c1, c2, c3 = st.columns(3)
        idade = c1.number_input("Idade", 16, 100, 30)
        renda = c2.number_input("Renda mensal líquida (R$)", 0.0, value=8000.0, step=500.0)
        despesas = c3.number_input("Despesas mensais (R$)", 0.0, value=5000.0, step=500.0)
        c1, c2, c3 = st.columns(3)
        tipo_renda = c1.selectbox(
            "Tipo de renda",
            ["clt", "servidor_publico", "autonomo", "empresario", "aposentado", "sem_renda_fixa"],
        )
        patrimonio = c2.number_input(
            "Patrimônio para investir (R$)", 0.0, value=30000.0, step=1000.0
        )
        aporte = c3.number_input("Aporte mensal (R$)", 0.0, value=1500.0, step=100.0)
        c1, c2, c3 = st.columns(3)
        perfil_risco = c1.selectbox(
            "Perfil de risco", ["conservador", "moderado", "arrojado"], index=1
        )
        experiencia = c2.selectbox("Experiência", ["iniciante", "intermediario", "avancado"])
        metodo = c3.selectbox("Método", ["modelo", "otimizado"])
        dependentes = c1.number_input("Dependentes", 0, 15, 0)
        dividas = c2.checkbox("Tenho dívidas caras (cartão, cheque especial)")

        st.markdown("**Objetivos**")
        objetivos_df = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "nome": "Entrada do apartamento",
                        "valor_alvo": 80000.0,
                        "prazo_anos": 5.0,
                        "prioridade": "alta",
                    },
                    {
                        "nome": "Aposentadoria",
                        "valor_alvo": 1500000.0,
                        "prazo_anos": 30.0,
                        "prioridade": "media",
                    },
                ]
            ),
            num_rows="dynamic",
            column_config={
                "prioridade": st.column_config.SelectboxColumn(options=["alta", "media", "baixa"])
            },
            use_container_width=True,
        )
        p1, p2, p3, p4 = st.columns(4)
        prefs = Preferencias(
            evitar_cripto=p1.checkbox("Evitar cripto"),
            evitar_exterior=p2.checkbox("Evitar exterior"),
            foco_renda_passiva=p3.checkbox("Foco em renda passiva"),
            prefere_isentos_ir=p4.checkbox("Preferir isentos de IR"),
        )
        enviado = st.form_submit_button("Montar carteira", type="primary")

    if not enviado:
        return
    objetivos = [
        Objetivo(**{k: row[k] for k in ("nome", "valor_alvo", "prazo_anos", "prioridade")})
        for row in objetivos_df.dropna().to_dict("records")
        if row.get("nome") and row.get("valor_alvo", 0) > 0 and row.get("prazo_anos", 0) > 0
    ]
    perfil = PerfilInvestidor(
        idade=idade,
        renda_mensal=renda,
        despesas_mensais=despesas,
        tipo_renda=tipo_renda,
        dependentes=dependentes,
        patrimonio_investido=patrimonio,
        aporte_mensal=aporte,
        possui_dividas_caras=dividas,
        perfil_risco=perfil_risco,
        experiencia=experiencia,
        objetivos=objetivos,
        preferencias=prefs,
        metodo=metodo,
    )
    carteira = montar_carteira(perfil).model_dump(mode="json")
    render_carteira(carteira)

    anos = max([o.prazo_anos for o in objetivos], default=20)
    w = model_portfolio(perfil_risco)
    sim = simulate(patrimonio, aporte, anos, portfolio_return(w), portfolio_volatility(w))
    render_simulacao(sim.model_dump())


st.title("📈 FinAI")
st.caption(
    "Assistente de investimentos com IA para o mercado brasileiro · ferramenta educacional, não é recomendação."
)
aba_chat, aba_offline = st.tabs(["💬 Assistente", "🧮 Planejador sem IA"])
with aba_chat:
    tab_assistente()
with aba_offline:
    tab_planejador()
