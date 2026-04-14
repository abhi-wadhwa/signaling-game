"""
Streamlit interactive UI for signaling game analysis.

Features:
  - Belief updating visualizer: prior -> signal -> posterior
  - Spence model: cost curves, wage schedules, equilibrium education
  - Game tree with belief annotations for Beer-Quiche
  - Crawford-Sobel partition visualization
  - Refinement filter: apply Intuitive Criterion, see which PBE survive
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.beer_quiche import BEER, FIGHT, NOT_FIGHT, QUICHE, TOUGH, WEAK, BeerQuicheGame
from src.core.crawford_sobel import CrawfordSobelModel
from src.core.d1_criterion import d1_criterion_filter
from src.core.intuitive_criterion import intuitive_criterion_filter
from src.core.spence import SpenceModel

st.set_page_config(
    page_title="Signaling Game Solver",
    page_icon="",
    layout="wide",
)

st.title("Signaling Game Equilibrium Solver")
st.markdown(
    "Analyze signaling games: Spence job market, Crawford-Sobel cheap talk, "
    "Beer-Quiche, with PBE enumeration and refinements."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Belief Updater",
    "Spence Model",
    "Crawford-Sobel",
    "Beer-Quiche & Refinements",
])


# ── Tab 1: Belief Updating Visualizer ──────────────────────────────────────
with tab1:
    st.header("Bayesian Belief Updating")
    bayes_formula = (
        r"$$\mu(\theta|m)="
        r"\frac{P(\theta)\sigma(m|\theta)}"
        r"{\sum_{\theta'}P(\theta')\sigma(m|\theta')}$$"
    )
    st.markdown(f"**Bayes' Rule** in signaling games:\n{bayes_formula}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Setup")
        num_types = st.slider("Number of types", 2, 5, 2, key="belief_ntypes")
        priors = []
        for i in range(num_types):
            p = st.slider(
                f"Prior P(type {i+1})",
                0.01, 0.99,
                1.0 / num_types,
                0.01,
                key=f"prior_{i}",
            )
            priors.append(p)

        # Normalize priors
        total = sum(priors)
        priors = [p / total for p in priors]

        st.markdown("**Signal likelihoods** (probability each type sends the observed signal):")
        likelihoods = []
        for i in range(num_types):
            ll = st.slider(
                f"sigma(signal | type {i+1})",
                0.0, 1.0, 0.5, 0.01,
                key=f"likelihood_{i}",
            )
            likelihoods.append(ll)

    with col2:
        st.subheader("Results")

        # Compute posterior
        numerators = [priors[i] * likelihoods[i] for i in range(num_types)]
        denom = sum(numerators)

        if denom > 1e-15:
            posteriors = [n / denom for n in numerators]
        else:
            posteriors = [0.0] * num_types
            st.warning("Signal has zero probability under all types (off-path).")

        # Bar chart comparison
        fig = go.Figure()
        type_labels = [f"Type {i+1}" for i in range(num_types)]

        fig.add_trace(go.Bar(
            name="Prior",
            x=type_labels,
            y=priors,
            marker_color="steelblue",
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="Posterior",
            x=type_labels,
            y=posteriors,
            marker_color="firebrick",
            opacity=0.7,
        ))

        fig.update_layout(
            barmode="group",
            title="Prior vs Posterior Beliefs",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show numerical values
        for i in range(num_types):
            st.metric(
                f"Type {i+1}",
                f"{posteriors[i]:.4f}",
                f"{posteriors[i] - priors[i]:+.4f}",
            )


# ── Tab 2: Spence Job Market Signaling ─────────────────────────────────────
with tab2:
    st.header("Spence Job Market Signaling Model")
    st.markdown(
        r"""
        Workers choose education $e \geq 0$, firms set wages $w(e) = E[\theta \mid e]$.
        Cost of education: $c(e, \theta) = e / \theta$ (single-crossing).

        **Separating equilibrium**: $e_L^* = 0$, $e_H^* = \theta_L(\theta_H - \theta_L)$.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        theta_l = st.slider("theta_L (low type)", 0.5, 3.0, 1.0, 0.1, key="sp_tl")
        theta_h = st.slider("theta_H (high type)", theta_l + 0.1, 5.0, 2.0, 0.1, key="sp_th")
        prob_h = st.slider("P(high type)", 0.01, 0.99, 0.5, 0.01, key="sp_ph")

    model = SpenceModel(theta_low=theta_l, theta_high=theta_h, prob_high=prob_h)
    sep_eq = model.separating_equilibrium()
    pool_eq = model.pooling_equilibrium()

    with col2:
        # Plot wage schedule and indifference curves
        fig = make_subplots(rows=1, cols=1)

        # Wage schedule (separating)
        ws_e = [pt[0] for pt in sep_eq.wage_schedule]
        ws_w = [pt[1] for pt in sep_eq.wage_schedule]
        fig.add_trace(go.Scatter(
            x=ws_e, y=ws_w, mode="lines", name="Wage schedule (sep.)",
            line=dict(color="navy", width=3),
        ))

        # Indifference curves
        e_star = sep_eq.education_levels["high"]
        e_max = max(e_star * 2.5, 3.0)

        # Low type IC through equilibrium point (e=0, w=theta_L)
        ic_low = model.indifference_curves(theta_l, sep_eq.payoffs["low"], (0, e_max))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in ic_low],
            y=[p[1] for p in ic_low],
            mode="lines", name=f"IC low (theta={theta_l})",
            line=dict(color="red", dash="dash"),
        ))

        # High type IC through equilibrium point
        ic_high = model.indifference_curves(theta_h, sep_eq.payoffs["high"], (0, e_max))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in ic_high],
            y=[p[1] for p in ic_high],
            mode="lines", name=f"IC high (theta={theta_h})",
            line=dict(color="green", dash="dash"),
        ))

        # Equilibrium points
        fig.add_trace(go.Scatter(
            x=[0, e_star],
            y=[sep_eq.wages["low"], sep_eq.wages["high"]],
            mode="markers+text",
            text=["Low type", "High type"],
            textposition="top right",
            marker=dict(size=12, color="black"),
            name="Equilibrium",
        ))

        fig.update_layout(
            title="Separating Equilibrium",
            xaxis_title="Education (e)",
            yaxis_title="Wage (w)",
            yaxis_range=[0, theta_h * 1.5],
            xaxis_range=[0, e_max],
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Separating Equilibrium")
        st.text(sep_eq.description)
    with col_b:
        st.subheader("Pooling Equilibrium")
        st.text(pool_eq.description)


# ── Tab 3: Crawford-Sobel Cheap Talk ───────────────────────────────────────
with tab3:
    st.header("Crawford-Sobel Cheap Talk Model")
    st.markdown(
        r"""
        State $\theta \sim U[0,1]$, sender bias $b \geq 0$.
        Partition $[0,1]$ into intervals; sender reveals which interval $\theta$ is in.
        Maximum partitions: $N^*(b)$.

        **Recursion**: $a_{i+1} = 2a_i - a_{i-1} + 4b$
        """
    )

    bias = st.slider("Sender bias (b)", 0.01, 0.5, 0.1, 0.01, key="cs_bias")
    cs_model = CrawfordSobelModel(bias=bias)
    n_star = cs_model.max_partitions()
    if isinstance(n_star, float):
        n_star_display = "infinity"
        n_star = 20
    else:
        n_star_display = str(n_star)

    st.metric("Maximum partitions N*(b)", n_star_display)

    equilibria = cs_model.all_partition_equilibria()

    # Partition visualization
    selected_n = st.slider(
        "Show N-partition equilibrium",
        1, min(n_star, 20), min(n_star, 20),
        key="cs_n",
    )

    eq = cs_model.partition_equilibrium(selected_n)
    if eq is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Draw partition intervals
            colors = [
                f"hsl({int(i * 360 / selected_n)}, 70%, 80%)"
                for i in range(selected_n)
            ]

            for i in range(selected_n):
                lo = eq.boundaries[i]
                hi = eq.boundaries[i + 1]
                action = eq.actions[i]

                fig.add_shape(
                    type="rect",
                    x0=lo, x1=hi, y0=0, y1=1,
                    fillcolor=colors[i],
                    opacity=0.5,
                    line=dict(color="black", width=1),
                )

                # Action line
                fig.add_shape(
                    type="line",
                    x0=lo, x1=hi, y0=action, y1=action,
                    line=dict(color="red", width=2, dash="dash"),
                )

                # Label
                mid = (lo + hi) / 2
                fig.add_annotation(
                    x=mid, y=action + 0.05,
                    text=f"y={action:.3f}",
                    showarrow=False,
                    font=dict(size=10),
                )

            # 45-degree line (full information)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Full information (y=theta)",
                line=dict(color="gray", dash="dot"),
            ))

            fig.update_layout(
                title=f"{selected_n}-Partition Equilibrium (b={bias})",
                xaxis_title="State (theta)",
                yaxis_title="Action (y)",
                xaxis_range=[0, 1],
                yaxis_range=[0, 1.1],
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Details")
            st.text(eq.description)

    # Welfare comparison across N
    if len(equilibria) > 1:
        st.subheader("Welfare Comparison")
        fig_welfare = go.Figure()
        ns = [eq.num_partitions for eq in equilibria]
        sender_eus = [eq.sender_eu for eq in equilibria]
        receiver_eus = [eq.receiver_eu for eq in equilibria]

        fig_welfare.add_trace(go.Bar(
            name="Sender EU", x=[str(n) for n in ns], y=sender_eus,
            marker_color="steelblue",
        ))
        fig_welfare.add_trace(go.Bar(
            name="Receiver EU", x=[str(n) for n in ns], y=receiver_eus,
            marker_color="firebrick",
        ))
        fig_welfare.update_layout(
            barmode="group",
            title="Expected Utility by Number of Partitions",
            xaxis_title="N (partitions)",
            yaxis_title="Expected Utility",
            height=350,
        )
        st.plotly_chart(fig_welfare, use_container_width=True)


# ── Tab 4: Beer-Quiche & Refinements ──────────────────────────────────────
with tab4:
    st.header("Beer-Quiche Signaling Game")
    st.markdown(
        r"""
        Two types (Tough/Weak), two signals (Beer/Quiche), two actions (Fight/Not Fight).

        Standard payoffs: Tough prefers Beer, Weak prefers Quiche; both prefer Not Fight.
        Receiver wants to Fight Weak, Not Fight Tough.

        **Intuitive Criterion** selects pooling-on-Beer when $P(\text{Tough})$ is high.
        """
    )

    prob_tough = st.slider(
        "P(Tough)", 0.01, 0.99, 0.9, 0.01, key="bq_pt"
    )

    bq = BeerQuicheGame(prob_tough=prob_tough)
    game = bq.to_signaling_game()

    all_pbe = bq.enumerate_all_pbe()

    st.subheader(f"All PBE ({len(all_pbe)} found)")

    type_names = {TOUGH: "Tough", WEAK: "Weak"}
    signal_names = {BEER: "Beer", QUICHE: "Quiche"}
    action_names = {FIGHT: "Fight", NOT_FIGHT: "Not Fight"}

    for i, pbe in enumerate(all_pbe):
        with st.expander(f"PBE {i+1}: {pbe.label} [{pbe.equilibrium_type}]"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Sender Strategy**")
                for t in [TOUGH, WEAK]:
                    sigs = pbe.get_sender_signal(t)
                    sig_str = ", ".join(
                        f"{signal_names[m]}: {p:.2f}" for m, p in sigs.items()
                    )
                    st.write(f"{type_names[t]}: {sig_str}")

            with col2:
                st.markdown("**Receiver Strategy**")
                for m in [BEER, QUICHE]:
                    acts = pbe.get_receiver_action(m)
                    if acts:
                        act_str = ", ".join(
                            f"{action_names[a]}: {p:.2f}" for a, p in acts.items()
                        )
                    else:
                        act_str = "(no actions)"
                    st.write(f"After {signal_names[m]}: {act_str}")

            with col3:
                st.markdown("**Beliefs**")
                for m in [BEER, QUICHE]:
                    belief = pbe.get_belief(m)
                    b_str = ", ".join(
                        f"{type_names[t]}: {p:.2f}" for t, p in belief.items()
                    )
                    st.write(f"mu(.|{signal_names[m]}): {b_str}")

            st.markdown("**Payoffs**")
            for t in [TOUGH, WEAK]:
                st.write(f"  {type_names[t]}: {pbe.sender_payoffs.get(t, 0):.2f}")

    # Refinements
    st.subheader("Equilibrium Refinements")

    col_ic, col_d1 = st.columns(2)

    with col_ic:
        st.markdown("**Intuitive Criterion (Cho-Kreps)**")
        ic_survivors = intuitive_criterion_filter(all_pbe, game)
        st.write(f"Survivors: {len(ic_survivors)} / {len(all_pbe)}")
        for pbe in ic_survivors:
            st.success(f"PASSES: {pbe.label}")
        for pbe in all_pbe:
            if pbe not in ic_survivors:
                st.error(f"FAILS: {pbe.label}")

    with col_d1:
        st.markdown("**D1 Criterion (Banks-Sobel)**")
        d1_survivors = d1_criterion_filter(all_pbe, game)
        st.write(f"Survivors: {len(d1_survivors)} / {len(all_pbe)}")
        for pbe in d1_survivors:
            st.success(f"PASSES: {pbe.label}")
        for pbe in all_pbe:
            if pbe not in d1_survivors:
                st.error(f"FAILS: {pbe.label}")

    # Game tree visualization
    st.subheader("Game Tree")

    fig_tree = go.Figure()

    # Layout: Nature at top, types below, signals below that, actions at bottom
    # Nature node
    fig_tree.add_trace(go.Scatter(
        x=[0.5], y=[4], mode="markers+text",
        text=["Nature"], textposition="top center",
        marker=dict(size=20, color="purple"),
        showlegend=False,
    ))

    # Type nodes
    type_x = {TOUGH: 0.2, WEAK: 0.8}
    for t, x in type_x.items():
        fig_tree.add_trace(go.Scatter(
            x=[x], y=[3], mode="markers+text",
            text=[f"{type_names[t]}\n(p={game.prior[t]:.2f})"],
            textposition="top center",
            marker=dict(size=15, color="blue"),
            showlegend=False,
        ))
        # Edge from Nature
        fig_tree.add_trace(go.Scatter(
            x=[0.5, x], y=[4, 3], mode="lines",
            line=dict(color="gray", width=1),
            showlegend=False,
        ))

    # Signal nodes
    signal_positions = {
        (TOUGH, BEER): 0.05, (TOUGH, QUICHE): 0.35,
        (WEAK, BEER): 0.65, (WEAK, QUICHE): 0.95,
    }

    for (t, m), x in signal_positions.items():
        fig_tree.add_trace(go.Scatter(
            x=[x], y=[2], mode="markers+text",
            text=[signal_names[m]], textposition="bottom center",
            marker=dict(size=12, color="orange"),
            showlegend=False,
        ))
        fig_tree.add_trace(go.Scatter(
            x=[type_x[t], x], y=[3, 2], mode="lines",
            line=dict(color="gray", width=1),
            showlegend=False,
        ))

    # Receiver information sets (dashed lines between same-signal nodes)
    fig_tree.add_shape(
        type="line",
        x0=signal_positions[(TOUGH, BEER)], y0=2,
        x1=signal_positions[(WEAK, BEER)], y1=2,
        line=dict(color="red", width=2, dash="dash"),
    )
    fig_tree.add_shape(
        type="line",
        x0=signal_positions[(TOUGH, QUICHE)], y0=2,
        x1=signal_positions[(WEAK, QUICHE)], y1=2,
        line=dict(color="red", width=2, dash="dash"),
    )

    # Action nodes
    for (t, m), x in signal_positions.items():
        for a, dx in [(FIGHT, -0.05), (NOT_FIGHT, 0.05)]:
            ax = x + dx
            fig_tree.add_trace(go.Scatter(
                x=[ax], y=[1], mode="markers+text",
                text=[action_names[a][:1]],
                textposition="bottom center",
                marker=dict(size=8, color="green"),
                showlegend=False,
            ))
            fig_tree.add_trace(go.Scatter(
                x=[x, ax], y=[2, 1], mode="lines",
                line=dict(color="gray", width=0.5),
                showlegend=False,
            ))

    fig_tree.update_layout(
        title="Beer-Quiche Game Tree",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_tree, use_container_width=True)
