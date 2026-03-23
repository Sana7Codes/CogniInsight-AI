"""
report_generator.py — Calls the Claude API to generate a personalised
cognitive performance report in French for a given user profile.

Model : claude-opus-4-6  (best reasoning quality)
Output: exactly 3 paragraphs in French, ≤ 350 words
"""

import os
from typing import Optional

import anthropic

# Label descriptions used to contextualise the cluster for the model
LABEL_DESCRIPTIONS = {
    "Focused": (
        "Profil Concentré — l'utilisateur affiche un temps de réaction rapide, "
        "une précision élevée et un faible taux d'erreur. Il est dans un état cognitif optimal."
    ),
    "Fatigué": (
        "Profil Fatigué — l'utilisateur affiche un temps de réaction lent, "
        "une faible précision et un taux d'erreur modéré à élevé. "
        "Des signes de fatigue mentale ou de sous-engagement sont présents."
    ),
    "Impulsif": (
        "Profil Impulsif — l'utilisateur répond très rapidement mais commet de nombreuses "
        "erreurs et présente une faible précision. Il agit avant d'avoir traité l'information "
        "complètement, signe d'impulsivité cognitive."
    ),
}


def generate_report(
    user_id: str,
    reaction_time_ms: float,
    accuracy_pct: float,
    error_rate: float,
    n_trials: int,
    cluster_label: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a personalised 3-paragraph French cognitive report using Claude.

    Parameters
    ----------
    user_id          : identifier shown in the report header
    reaction_time_ms : mean reaction time in milliseconds
    accuracy_pct     : mean accuracy percentage (0–100)
    error_rate       : mean error rate percentage (0–100)
    n_trials         : total number of trials performed
    cluster_label    : one of "Focused", "Fatigué", "Impulsif"
    api_key          : optional override; falls back to ANTHROPIC_API_KEY env var

    Returns
    -------
    str — the report text (3 paragraphs)

    Raises
    ------
    EnvironmentError : if no API key is available
    anthropic.APIError : on API failures
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your environment or .env file."
        )

    client = anthropic.Anthropic(api_key=resolved_key)

    cluster_context = LABEL_DESCRIPTIONS.get(
        cluster_label,
        f"Profil {cluster_label} — description non disponible."
    )

    # Compute a quick percentile note for context enrichment
    rt_note = (
        "rapide (< 270 ms)" if reaction_time_ms < 270
        else "lent (> 380 ms)" if reaction_time_ms > 380
        else "dans la moyenne (270–380 ms)"
    )
    acc_note = (
        "excellente (> 90 %)" if accuracy_pct > 90
        else "faible (< 75 %)" if accuracy_pct < 75
        else "modérée (75–90 %)"
    )

    prompt = f"""Tu es un expert en neurosciences cognitives et en psychologie de la performance.
Voici les résultats de tests cognitifs informatisés pour l'utilisateur **{user_id}** :

| Indicateur | Valeur |
|---|---|
| Temps de réaction moyen | {reaction_time_ms:.1f} ms ({rt_note}) |
| Précision moyenne | {accuracy_pct:.1f} % ({acc_note}) |
| Taux d'erreur moyen | {error_rate:.1f} % |
| Nombre d'essais | {n_trials} |

**Profil cognitif détecté (clustering KMeans) : {cluster_label}**
{cluster_context}

Rédige un rapport cognitif personnalisé en FRANÇAIS composé d'exactement 3 paragraphes :

1. **Analyse des données** : Décris ce que les chiffres exacts révèlent sur l'état cognitif de l'utilisateur. Cite les valeurs numériques. Sois précis et factuel.

2. **Facteurs explicatifs** : Identifie les causes probables de ce profil (fatigue, stress, impulsivité, stratégies cognitives inefficaces, etc.). Base-toi sur les métriques spécifiques. Évite le vague.

3. **Recommandations concrètes** : Donne 3 à 4 actions quotidiennes très concrètes pour améliorer ce profil. Pas de conseils génériques : adapte chaque recommandation aux valeurs mesurées.

Contraintes :
- Maximum 350 mots au total.
- Langue : français courant mais professionnel.
- Chaque paragraphe commence par un titre en gras (ex: **Analyse**, **Facteurs**, **Recommandations**).
- Ne répète pas le tableau de données.
- Sois direct, actionnable, et évite toute formulation banale.
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=700,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text block from response (thinking blocks precede text blocks)
    for block in response.content:
        if block.type == "text":
            return block.text.strip()

    return "Erreur : aucun texte généré par le modèle."
