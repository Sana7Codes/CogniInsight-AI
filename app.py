"""
app.py — CogniInsight AI  ·  Main Streamlit dashboard.

Features:
  1. Upload a CSV of cognitive test sessions OR load built-in sample data.
  2. KMeans clustering → 3 cognitive profiles (Focused / Fatigué / Impulsif).
  3. PCA 2D scatter plot of clusters (Plotly).
  4. Per-user session trend charts (Plotly line charts).
  5. Claude-generated personalised French report.
  6. Download report as PDF.

Run:
  streamlit run app.py
"""

import io
import os

import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering import CLUSTER_FEATURES, get_cluster_stats, run_clustering
from pdf_export import create_pdf_report
from report_generator import generate_report
from sample_data import generate_sample_data, get_user_aggregate

# ------------------------------------------------------------------ #
#  Page config — must be the FIRST Streamlit call                     #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="CogniInsight AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Dark-theme custom CSS                                              #
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    /* Cards / metric boxes */
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Headings */
    h1, h2, h3 { color: #f1f5f9 !important; }

    /* DataFrames */
    .dataframe { background-color: #1e293b !important; color: #e2e8f0 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Selectbox / inputs */
    .stSelectbox > div > div { background-color: #1e293b; color: #e2e8f0; }
    .stTextInput > div > div > input { background-color: #1e293b; color: #e2e8f0; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { background-color: #1e293b; color: #94a3b8; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }

    /* Expander */
    .streamlit-expanderHeader { background-color: #1e293b; color: #cbd5e1; border-radius: 8px; }

    /* Dividers */
    hr { border-color: #334155; }

    /* Info / warning boxes */
    .stAlert { background-color: #1e293b; border-left-color: #3b82f6; }

    /* Report text area */
    .report-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 20px;
        font-size: 15px;
        line-height: 1.7;
        color: #e2e8f0;
        white-space: pre-wrap;
    }

    /* Profile badge */
    .badge-focused { background:#166534; color:#86efac; border-radius:6px; padding:4px 12px; font-weight:700; }
    .badge-fatigue  { background:#854d0e; color:#fde68a; border-radius:6px; padding:4px 12px; font-weight:700; }
    .badge-impulsif{ background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:4px 12px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
#  Colour maps                                                        #
# ------------------------------------------------------------------ #
PROFILE_COLORS = {
    "Focused":  "#22c55e",
    "Fatigué":  "#eab308",
    "Impulsif": "#ef4444",
}

PROFILE_BADGE_HTML = {
    "Focused":  '<span class="badge-focused">● Focused</span>',
    "Fatigué":  '<span class="badge-fatigue">◆ Fatigué</span>',
    "Impulsif": '<span class="badge-impulsif">▲ Impulsif</span>',
}


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def plotly_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply a consistent dark theme to any Plotly figure."""
    fig.update_layout(
        title={"text": title, "font": {"color": "#f1f5f9", "size": 15}},
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        legend={"bgcolor": "#1e293b", "bordercolor": "#334155", "borderwidth": 1},
        xaxis={"gridcolor": "#334155", "linecolor": "#334155"},
        yaxis={"gridcolor": "#334155", "linecolor": "#334155"},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return generate_sample_data()


@st.cache_data
def compute_clustering(df_json: str):
    """Cache-aware wrapper around run_clustering (accepts JSON string for hashing)."""
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return run_clustering(df)


# ------------------------------------------------------------------ #
#  Sidebar                                                            #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/brain.png",
        width=64,
    )
    st.title("CogniInsight AI")
    st.caption("Analyse cognitive par Intelligence Artificielle")
    st.divider()

    st.subheader("📂 Source de données")
    data_source = st.radio(
        "Choisir la source",
        ["📊 Données exemple", "📤 Importer un CSV"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("⚙️ Paramètres IA")
    api_key_input = st.text_input(
        "Clé API Anthropic",
        type="password",
        placeholder="sk-ant-...",
        help="Chargée depuis ANTHROPIC_API_KEY si vide",
    )

    st.divider()
    st.subheader("ℹ️ À propos")
    st.markdown(
        """
        **CogniInsight AI** utilise :
        - `scikit-learn` — clustering KMeans
        - `Plotly` — visualisations interactives
        - `Claude claude-opus-4-6` — rapports en français
        - `fpdf2` — export PDF

        [GitHub](https://github.com) · [Hugging Face](https://huggingface.co)
        """
    )


# ------------------------------------------------------------------ #
#  Data loading                                                       #
# ------------------------------------------------------------------ #
raw_df: pd.DataFrame | None = None

if data_source == "📊 Données exemple":
    raw_df = load_sample_data()
    st.toast("✅ Données exemple chargées (20 utilisateurs · 50 sessions)", icon="🧪")
else:
    uploaded_file = st.file_uploader(
        "Déposer un fichier CSV",
        type=["csv"],
        help="Colonnes requises : reaction_time_ms, accuracy_pct, error_rate, n_trials, user_id",
    )
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            required_cols = {"user_id", "reaction_time_ms", "accuracy_pct", "error_rate"}
            missing = required_cols - set(raw_df.columns)
            if missing:
                st.error(f"❌ Colonnes manquantes : {missing}")
                raw_df = None
            else:
                st.success(f"✅ Fichier chargé : {len(raw_df)} lignes · {raw_df['user_id'].nunique()} utilisateurs")
        except Exception as e:
            st.error(f"❌ Erreur de lecture : {e}")

# ------------------------------------------------------------------ #
#  Main content — only shown once data is loaded                      #
# ------------------------------------------------------------------ #
if raw_df is None:
    # Landing page
    st.markdown(
        """
        <div style="text-align:center; padding: 80px 0 40px 0;">
            <h1 style="font-size:3rem; background: linear-gradient(135deg,#3b82f6,#8b5cf6);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🧠 CogniInsight AI
            </h1>
            <p style="font-size:1.2rem; color:#94a3b8; max-width:600px; margin:auto;">
                Plateforme d'analyse cognitive alimentée par Intelligence Artificielle.
                Chargez vos données de tests cognitifs pour obtenir des profils personnalisés
                et des recommandations générées par Claude.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for col, icon, label, desc in [
        (c1, "🎯", "Clustering KMeans", "3 profils cognitifs automatiquement identifiés"),
        (c2, "📈", "Visualisation PCA", "Scatter plot 2D interactif des clusters"),
        (c3, "🤖", "Rapport Claude AI", "Analyse personnalisée en français par claude-opus-4-6"),
    ]:
        col.markdown(
            f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
                            padding:24px;text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <h3 style="color:#f1f5f9;margin:8px 0 4px;">{label}</h3>
                <p style="color:#94a3b8;font-size:0.9rem;">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    st.stop()

# ------------------------------------------------------------------ #
#  Aggregate per-user and run clustering                              #
# ------------------------------------------------------------------ #
# If CSV has session_number column, aggregate per user
if "session_number" in raw_df.columns:
    user_df = get_user_aggregate(raw_df)
else:
    # Assume one row per user
    user_df = raw_df[["user_id", "reaction_time_ms", "accuracy_pct", "error_rate", "n_trials"]].copy()
    user_df["n_sessions"] = 1

# Ensure n_clusters doesn't exceed number of users
n_users = len(user_df)
if n_users < 3:
    st.error("❌ Au moins 3 utilisateurs sont nécessaires pour le clustering.")
    st.stop()

cluster_result = compute_clustering(user_df.to_json(orient="records"))
df_clustered = cluster_result["df_clustered"]
pca_components = cluster_result["pca_components"]
pca_var = cluster_result["pca_variance"]

# Attach PCA coords to clustered df
df_vis = df_clustered.copy()
df_vis["PC1"] = pca_components[:, 0]
df_vis["PC2"] = pca_components[:, 1]

# ------------------------------------------------------------------ #
#  KPI metrics row                                                     #
# ------------------------------------------------------------------ #
st.markdown("## 📊 Vue d'ensemble")
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("👥 Utilisateurs", n_users)
k2.metric("⚡ RT moyen", f"{user_df['reaction_time_ms'].mean():.0f} ms")
k3.metric("🎯 Précision moy.", f"{user_df['accuracy_pct'].mean():.1f} %")
k4.metric("❌ Erreur moy.", f"{user_df['error_rate'].mean():.1f} %")

# Most common cluster
top_cluster = df_clustered["cluster_label"].value_counts().idxmax()
k5.metric("🏆 Profil dominant", top_cluster)

st.divider()

# ------------------------------------------------------------------ #
#  Tabs                                                               #
# ------------------------------------------------------------------ #
tab_cluster, tab_trends, tab_report = st.tabs(
    ["🗺️ Clustering & PCA", "📈 Tendances par utilisateur", "🤖 Rapport IA"]
)

# ======================================================= #
# TAB 1 — Clustering & PCA                                #
# ======================================================= #
with tab_cluster:
    col_pca, col_stats = st.columns([3, 2])

    with col_pca:
        st.subheader("Scatter PCA — Profils cognitifs")

        fig_pca = px.scatter(
            df_vis,
            x="PC1",
            y="PC2",
            color="cluster_label",
            color_discrete_map=PROFILE_COLORS,
            hover_data={
                "user_id": True,
                "reaction_time_ms": ":.1f",
                "accuracy_pct": ":.1f",
                "error_rate": ":.1f",
                "PC1": False,
                "PC2": False,
            },
            labels={
                "PC1": f"PC1 ({pca_var[0]} % variance)",
                "PC2": f"PC2 ({pca_var[1]} % variance)",
                "cluster_label": "Profil",
            },
            symbol="cluster_label",
            size_max=12,
        )
        fig_pca.update_traces(marker={"size": 11, "opacity": 0.85, "line": {"width": 1, "color": "#334155"}})
        plotly_dark_layout(fig_pca, f"PCA 2D · Variance expliquée : {pca_var[0] + pca_var[1]:.1f} %")
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_stats:
        st.subheader("Statistiques par profil")
        stats_df = get_cluster_stats(df_clustered)

        for _, row in stats_df.iterrows():
            label = row["cluster_label"]
            color = PROFILE_COLORS[label]
            n = int(row["n_users"])
            rt = row["reaction_time_ms_mean"]
            acc = row["accuracy_pct_mean"]
            err = row["error_rate_mean"]

            st.markdown(
                f"""<div style="background:#1e293b;border-left:4px solid {color};
                                border-radius:8px;padding:14px;margin-bottom:12px;">
                    <div style="font-size:1.1rem;font-weight:700;color:{color};margin-bottom:8px;">
                        {label} — {n} utilisateur(s)
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:0.9rem;">
                        <div><span style="color:#94a3b8;">Réaction</span><br>
                             <b style="color:#f1f5f9;">{rt:.0f} ms</b></div>
                        <div><span style="color:#94a3b8;">Précision</span><br>
                             <b style="color:#f1f5f9;">{acc:.1f} %</b></div>
                        <div><span style="color:#94a3b8;">Erreur</span><br>
                             <b style="color:#f1f5f9;">{err:.1f} %</b></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # Feature importance bar chart (mean values per cluster)
        st.subheader("Comparaison des features")
        feat_data = []
        for _, row in stats_df.iterrows():
            for feat, label_col in zip(
                ["reaction_time_ms", "accuracy_pct", "error_rate"],
                ["reaction_time_ms_mean", "accuracy_pct_mean", "error_rate_mean"],
            ):
                feat_data.append({"Profil": row["cluster_label"], "Feature": feat, "Valeur": row[label_col]})

        feat_df = pd.DataFrame(feat_data)
        fig_bar = px.bar(
            feat_df,
            x="Feature",
            y="Valeur",
            color="Profil",
            barmode="group",
            color_discrete_map=PROFILE_COLORS,
        )
        plotly_dark_layout(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)


# ======================================================= #
# TAB 2 — Per-user trends                                 #
# ======================================================= #
with tab_trends:
    st.subheader("Évolution par session — Sélectionnez un utilisateur")

    # Only show users with multiple sessions
    if "session_number" in raw_df.columns:
        multi_session_users = raw_df.groupby("user_id").size()
        multi_session_users = multi_session_users[multi_session_users > 1].index.tolist()
        if not multi_session_users:
            multi_session_users = raw_df["user_id"].unique().tolist()
    else:
        multi_session_users = raw_df["user_id"].unique().tolist()

    selected_user = st.selectbox("Utilisateur", sorted(multi_session_users))

    if selected_user:
        # Filter user sessions
        if "session_number" in raw_df.columns:
            user_sessions = raw_df[raw_df["user_id"] == selected_user].sort_values("session_number")
        else:
            user_sessions = raw_df[raw_df["user_id"] == selected_user]
            user_sessions["session_number"] = range(1, len(user_sessions) + 1)

        # Get this user's cluster label
        user_cluster = df_clustered[df_clustered["user_id"] == selected_user]["cluster_label"].values
        user_cluster_label = user_cluster[0] if len(user_cluster) > 0 else "N/A"
        cluster_color = PROFILE_COLORS.get(user_cluster_label, "#94a3b8")

        # Header info
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Profil cognitif", user_cluster_label)
        col_info2.metric("Sessions", len(user_sessions))
        col_info3.metric("Tests total", int(user_sessions["n_trials"].sum()))

        st.markdown(
            f'<div style="margin:8px 0 16px;">Profil : {PROFILE_BADGE_HTML.get(user_cluster_label, user_cluster_label)}</div>',
            unsafe_allow_html=True,
        )

        # Multi-metric line chart
        x_col = "session_number"
        if "session_date" in user_sessions.columns:
            x_col = "session_date"
            # Sort by date
            user_sessions = user_sessions.sort_values("session_date")

        fig_trend = go.Figure()
        metrics_config = [
            ("reaction_time_ms", "Temps de réaction (ms)", "#3b82f6", True),
            ("accuracy_pct", "Précision (%)", "#22c55e", False),
            ("error_rate", "Taux d'erreur (%)", "#ef4444", False),
        ]
        for col, name, color, secondary in metrics_config:
            if col in user_sessions.columns:
                yaxis = "y2" if secondary else "y"
                fig_trend.add_trace(
                    go.Scatter(
                        x=user_sessions[x_col],
                        y=user_sessions[col],
                        mode="lines+markers",
                        name=name,
                        line={"color": color, "width": 2.5},
                        marker={"size": 8},
                        yaxis=yaxis,
                    )
                )

        fig_trend.update_layout(
            yaxis={"title": "Précision / Erreur (%)", "gridcolor": "#334155"},
            yaxis2={
                "title": "Temps de réaction (ms)",
                "overlaying": "y",
                "side": "right",
                "gridcolor": "#334155",
            },
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font={"color": "#e2e8f0"},
            legend={"bgcolor": "#1e293b"},
            hovermode="x unified",
            margin={"l": 40, "r": 60, "t": 40, "b": 40},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Radar / spider chart of this user vs cluster mean
        user_agg = df_clustered[df_clustered["user_id"] == selected_user]
        if len(user_agg) > 0:
            cluster_mean = df_clustered[df_clustered["cluster_label"] == user_cluster_label][CLUSTER_FEATURES].mean()
            user_vals = user_agg[CLUSTER_FEATURES].values[0]

            categories = ["Réaction (ms)", "Précision (%)", "Erreur (%)"]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(user_vals) + [user_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=selected_user,
                line_color=cluster_color,
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(cluster_mean.values) + [cluster_mean.values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"Moy. {user_cluster_label}",
                line_color="#64748b",
                opacity=0.5,
            ))
            fig_radar.update_layout(
                polar={
                    "radialaxis": {"visible": True, "gridcolor": "#334155"},
                    "bgcolor": "#0f172a",
                    "angularaxis": {"gridcolor": "#334155"},
                },
                paper_bgcolor="#1e293b",
                font={"color": "#e2e8f0"},
                title={"text": "Profil vs Moyenne du cluster", "font": {"color": "#f1f5f9"}},
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ======================================================= #
# TAB 3 — AI Report                                       #
# ======================================================= #
with tab_report:
    st.subheader("🤖 Rapport cognitif personnalisé par Claude")
    st.markdown(
        "Sélectionnez un utilisateur et générez une analyse en français, "
        "incluant une interprétation des données, des causes probables et des recommandations concrètes."
    )

    col_select, col_gen = st.columns([2, 1])

    with col_select:
        report_user = st.selectbox(
            "Utilisateur",
            sorted(df_clustered["user_id"].tolist()),
            key="report_user_select",
        )

    with col_gen:
        st.write("")
        st.write("")
        generate_btn = st.button("🧠 Générer le rapport", use_container_width=True)

    # User metrics preview
    if report_user:
        user_row = df_clustered[df_clustered["user_id"] == report_user].iloc[0]
        r_label = user_row["cluster_label"]
        r_color = PROFILE_COLORS[r_label]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RT moyen", f"{user_row['reaction_time_ms']:.0f} ms")
        c2.metric("Précision", f"{user_row['accuracy_pct']:.1f} %")
        c3.metric("Erreur", f"{user_row['error_rate']:.1f} %")
        c4.metric("Essais", int(user_row["n_trials"]))
        c5.markdown(
            f'<div style="margin-top:24px;">{PROFILE_BADGE_HTML[r_label]}</div>',
            unsafe_allow_html=True,
        )

    # Report generation
    if generate_btn and report_user:
        user_row = df_clustered[df_clustered["user_id"] == report_user].iloc[0]

        # Resolve API key
        key = api_key_input.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            st.error("❌ Clé API manquante. Ajoutez-la dans la barre latérale ou définissez ANTHROPIC_API_KEY.")
        else:
            with st.spinner("Claude génère votre rapport... ⏳"):
                try:
                    report_text = generate_report(
                        user_id=report_user,
                        reaction_time_ms=float(user_row["reaction_time_ms"]),
                        accuracy_pct=float(user_row["accuracy_pct"]),
                        error_rate=float(user_row["error_rate"]),
                        n_trials=int(user_row["n_trials"]),
                        cluster_label=user_row["cluster_label"],
                        api_key=key,
                    )

                    # Store in session state for PDF export
                    st.session_state["report_text"] = report_text
                    st.session_state["report_user"] = report_user
                    st.session_state["report_user_row"] = user_row.to_dict()

                    st.success("✅ Rapport généré avec succès !")

                except EnvironmentError as e:
                    st.error(f"❌ Clé API : {e}")
                except anthropic.AuthenticationError:
                    st.error("❌ Clé API invalide. Vérifiez votre clé Anthropic.")
                except anthropic.RateLimitError:
                    st.error("⚠️ Limite d'API atteinte. Réessayez dans quelques instants.")
                except anthropic.APIConnectionError:
                    st.error("🌐 Erreur réseau. Vérifiez votre connexion Internet.")
                except anthropic.APIStatusError as e:
                    st.error(f"❌ Erreur API ({e.status_code}) : {e.message}")
                except Exception as e:
                    st.error(f"❌ Erreur inattendue : {e}")

    # Display stored report
    if "report_text" in st.session_state:
        rtext = st.session_state["report_text"]
        ruser = st.session_state["report_user"]
        rrow  = st.session_state["report_user_row"]

        st.markdown(f"### Rapport pour `{ruser}`")
        st.markdown(
            f'<div class="report-box">{rtext}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # PDF export
        st.subheader("📄 Télécharger le rapport")
        col_dl1, col_dl2 = st.columns([1, 2])

        with col_dl1:
            if st.button("🖨️ Préparer le PDF", use_container_width=True):
                with st.spinner("Génération du PDF..."):
                    try:
                        pdf_bytes = create_pdf_report(
                            user_id=rrow["user_id"],
                            reaction_time_ms=float(rrow["reaction_time_ms"]),
                            accuracy_pct=float(rrow["accuracy_pct"]),
                            error_rate=float(rrow["error_rate"]),
                            n_trials=int(rrow["n_trials"]),
                            cluster_label=rrow["cluster_label"],
                            report_text=rtext,
                        )
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.success("✅ PDF prêt !")
                    except Exception as e:
                        st.error(f"❌ Erreur PDF : {e}")

        with col_dl2:
            if "pdf_bytes" in st.session_state:
                st.download_button(
                    label="⬇️ Télécharger le PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=f"cogniinsight_{rrow['user_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
