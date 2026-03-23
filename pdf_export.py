"""
pdf_export.py — Generates a formatted PDF report using fpdf2.

Layout:
  - Header with project branding and user ID
  - Metrics summary table
  - Cluster profile badge
  - Full 3-paragraph French report text
  - Footer with generation date
"""

import io
from datetime import datetime
from fpdf import FPDF, XPos, YPos

# Brand colours (approximated for PDF)
COLOR_PRIMARY = (30, 120, 200)    # blue
COLOR_SECONDARY = (60, 60, 60)    # dark grey
COLOR_ACCENT_FOCUSED = (34, 197, 94)    # green
COLOR_ACCENT_FATIGUED = (234, 179, 8)   # amber
COLOR_ACCENT_IMPULSIVE = (239, 68, 68)  # red
COLOR_BG_HEADER = (15, 23, 42)          # near-black
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GREY = (245, 245, 245)

PROFILE_COLORS = {
    "Focused": COLOR_ACCENT_FOCUSED,
    "Fatigué": COLOR_ACCENT_FATIGUED,
    "Impulsif": COLOR_ACCENT_IMPULSIVE,
}

PROFILE_EMOJI = {
    "Focused": "●",
    "Fatigué": "◆",
    "Impulsif": "▲",
}


def sanitize(text: str) -> str:
    """
    Replace Unicode characters unsupported by Helvetica with ASCII equivalents.
    Covers all common Claude-generated French typographic characters.
    """
    return (
        text
        .replace("\u2014", "-")   # em dash —
        .replace("\u2013", "-")   # en dash –
        .replace("\u2019", "'")   # right single quote '
        .replace("\u2018", "'")   # left single quote '
        .replace("\u201c", '"')   # left double quote "
        .replace("\u201d", '"')   # right double quote "
        .replace("\u2026", "...")  # ellipsis …
        .replace("\u00b0", " deg")  # degree °
        .replace("\u2022", "-")   # bullet •
        .replace("\u25cf", "*")   # filled circle ●
        .replace("\u25c6", "*")   # filled diamond ◆
        .replace("\u25b2", "*")   # filled triangle ▲
        .replace("\u00e9", "e")   # é  -- only needed if font truly can't handle Latin-1
        # Note: Helvetica supports full Latin-1 (accented French chars),
        # so we only strip chars outside that range above.
    )


class CognitivePDF(FPDF):
    """Custom FPDF subclass with branding helpers."""

    def header(self):
        self.set_fill_color(*COLOR_BG_HEADER)
        self.rect(0, 0, 210, 22, style="F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*COLOR_WHITE)
        self.set_xy(10, 6)
        self.cell(
            0, 10,
            sanitize("CogniInsight AI - Rapport de Performance Cognitive"),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.cell(
            0, 10,
            sanitize(f"Genere le {now}  |  CogniInsight AI  |  Page {self.page_no()}"),
            align="C"
        )


def _section_title(pdf: CognitivePDF, title: str) -> None:
    """Draw a coloured left-border section title."""
    pdf.set_fill_color(*COLOR_PRIMARY)
    pdf.rect(10, pdf.get_y(), 3, 7, style="F")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(0, 7, sanitize(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def _metrics_table(
    pdf: CognitivePDF,
    reaction_time_ms: float,
    accuracy_pct: float,
    error_rate: float,
    n_trials: int,
) -> None:
    """Draw a 2-column metrics summary table."""
    rows = [
        ("Temps de reaction moyen", f"{reaction_time_ms:.1f} ms"),
        ("Precision moyenne", f"{accuracy_pct:.1f} %"),
        ("Taux d'erreur moyen", f"{error_rate:.1f} %"),
        ("Nombre d'essais", str(n_trials)),
    ]

    col_w = [90, 90]
    row_h = 8

    pdf.set_fill_color(*COLOR_PRIMARY)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(10)
    pdf.cell(col_w[0], row_h, "Indicateur", border=0, fill=True)
    pdf.cell(col_w[1], row_h, "Valeur", border=0, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.set_font("Helvetica", "", 10)
    for i, (label, value) in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*COLOR_LIGHT_GREY) if fill else pdf.set_fill_color(*COLOR_WHITE)
        pdf.set_x(10)
        pdf.cell(col_w[0], row_h, sanitize(label), border=0, fill=True)
        pdf.cell(col_w[1], row_h, sanitize(value), border=0, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)


def _cluster_badge(pdf: CognitivePDF, cluster_label: str) -> None:
    """Draw a coloured pill/badge showing the detected profile."""
    color = PROFILE_COLORS.get(cluster_label, COLOR_SECONDARY)

    y = pdf.get_y()
    pdf.set_fill_color(*color)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font("Helvetica", "B", 11)
    badge_text = sanitize(f"  Profil detecte : {cluster_label}  ")
    badge_w = pdf.get_string_width(badge_text) + 6
    pdf.set_xy(10, y)
    pdf.cell(badge_w, 10, badge_text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def _report_body(pdf: CognitivePDF, report_text: str) -> None:
    """Render multi-paragraph report text with bold paragraph titles."""
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.set_font("Helvetica", "", 10)

    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]

    for para in paragraphs:
        if para.startswith("**") and "**" in para[2:]:
            end_bold = para.index("**", 2)
            title_text = sanitize(para[2:end_bold].strip())
            rest_text = sanitize(para[end_bold + 2:].strip(" :"))

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(10)
            pdf.cell(0, 6, title_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(10)
            pdf.multi_cell(0, 5.5, rest_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_x(10)
            pdf.multi_cell(0, 5.5, sanitize(para), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(3)


def create_pdf_report(
    user_id: str,
    reaction_time_ms: float,
    accuracy_pct: float,
    error_rate: float,
    n_trials: int,
    cluster_label: str,
    report_text: str,
) -> bytes:
    """
    Build the PDF and return it as raw bytes (suitable for st.download_button).
    """
    pdf = CognitivePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*COLOR_BG_HEADER)
    pdf.set_x(10)
    pdf.cell(0, 8, sanitize(f"Utilisateur : {user_id}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    _section_title(pdf, "Metriques cognitives")
    _metrics_table(pdf, reaction_time_ms, accuracy_pct, error_rate, n_trials)

    _section_title(pdf, "Classification du profil")
    _cluster_badge(pdf, cluster_label)

    _section_title(pdf, "Analyse personnalisee (generee par IA)")
    _report_body(pdf, report_text)

    return bytes(pdf.output())