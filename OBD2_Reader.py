# -*- coding: utf-8 -*-
"""
OBD2 Analyse - Car Scanner ELM OBD2 (Stanislav Svistunov)
==========================================================
Streamlit-app for interaktiv visualisering og helseanalyse av OBD2-data.

Kjoer med:
    streamlit run obd2_app.py

Avhengigheter:
    pip install streamlit plotly pandas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# SIDEKONFIGURASJON
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="OBD2 Analyse",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# SIGNALDEFINISJONER
# -----------------------------------------------------------------------------

SIGNALER = {
    "Vehicle speed":                    ("Hastighet",           "km/h",  "#1565C0"),
    "Engine RPM":                       ("Motorturtall",        "rpm",   "#B71C1C"),
    "Calculated engine load value":     ("Motorlast",           "%",     "#E65100"),
    "Calculated instant fuel rate":     ("Drivstofforbruk",     "L/h",   "#1B5E20"),
    "MAF air flow rate":                ("Luftmasserate",       "g/s",   "#4A148C"),
    "Vehicle acceleration":             ("Akselerasjon",        "g",     "#006064"),
    "Throttle position":                ("Gasspårag",           "%",     "#F57F17"),
    "Engine coolant temperature":       ("Kjølevæske",          "°C",    "#880E4F"),
    "Intake air temperature":           ("Innsugsluft",         "°C",    "#BF360C"),
    "Instant engine power (based on fuel consumption)": ("Motoreffekt", "hp", "#37474F"),
    "Power from MAF":                   ("Effekt (MAF)",        "hp",    "#5D4037"),
    "Calculated boost":                 ("Boost",               "psi",   "#0277BD"),
    "Intake manifold absolute pressure":("Innsugstrykk",        "psi",   "#00695C"),
    "Oxygen sensor 1 Wide Range Voltage": ("O2-sensor",         "V",     "#6A1B9A"),
    "Fuel rail press.":                 ("Drivstofftrykk",      "psi",   "#558B2F"),
    "OBD Module Voltage":               ("Batterispenning",     "V",     "#F9A825"),
    "Distance travelled":               ("Distanse",            "km",    "#37474F"),
    "Fuel used":                        ("Drivstoff brukt",     "L",     "#2E7D32"),
    "Average speed":                    ("Snittfart",           "km/h",  "#1976D2"),
}

SKJUL = {
    "Fuel used (Today)", "Fuel used (Week)", "Fuel used (total)",
    "Fuel used price", "Fuel used price (Today)",
    "Fuel used price (total)", "Fuel used price (Week)",
    "Distance travelled (Today)", "Distance travelled (Week)",
    "Distance travelled (total)", "Engine RPM x1000",
    "Distance traveled with MIL on",
}

# -----------------------------------------------------------------------------
# DATAINNLESING
# -----------------------------------------------------------------------------

@st.cache_data
def importer_car_scanner(filinnhold: bytes) -> pd.DataFrame:
    import io
    df_raa = pd.read_csv(
        io.BytesIO(filinnhold),
        sep=";", quotechar='"', dtype=str,
        skip_blank_lines=True, encoding="utf-8-sig",
    )
    df_raa.columns = ["SECONDS", "PID", "VALUE", "UNITS", *df_raa.columns[4:]]
    df_raa = df_raa[["SECONDS", "PID", "VALUE"]].dropna(subset=["PID"])
    df_raa["SECONDS"] = pd.to_numeric(df_raa["SECONDS"], errors="coerce")
    df_raa["VALUE"]   = pd.to_numeric(df_raa["VALUE"],   errors="coerce")
    df_raa = df_raa.dropna(subset=["SECONDS", "VALUE"])
    df_bred = df_raa.pivot_table(
        index="SECONDS", columns="PID", values="VALUE", aggfunc="mean"
    )
    df_bred.index = df_bred.index - df_bred.index.min()
    df_bred.index.name = "tid_s"
    return df_bred


def hent_signal(df: pd.DataFrame, pid: str):
    if pid not in df.columns:
        return None
    return df[pid].ffill().bfill()

# -----------------------------------------------------------------------------
# PLOTTEFUNKSJONER
# -----------------------------------------------------------------------------

def hex_til_rgba(hex_farge: str, alpha: float = 0.12) -> str:
    h = hex_farge.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def lag_tidsserie(df: pd.DataFrame, valgte_pid: list) -> go.Figure:
    n = len(valgte_pid)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=[SIGNALER.get(p, (p,))[0] for p in valgte_pid],
    )
    for i, pid in enumerate(valgte_pid, start=1):
        navn, enhet, farge = SIGNALER.get(pid, (pid, "", "#607D8B"))
        serie = hent_signal(df, pid)
        if serie is None:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=serie,
            name=f"{navn} ({enhet})",
            line=dict(color=farge, width=1.5),
            fill="tozeroy", fillcolor=hex_til_rgba(farge),
            hovertemplate=f"<b>{navn}</b><br>Tid: %{{x:.1f}} s<br>%{{y:.2f}} {enhet}<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(title_text=enhet, row=i, col=1, title_font_size=10)
    fig.update_xaxes(title_text="Tid (sekunder fra start)", row=n, col=1)
    fig.update_layout(
        height=max(300, 260 * n), showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40), hovermode="x unified",
    )
    return fig


def lag_dashboard(df: pd.DataFrame) -> go.Figure:
    paneler = [
        ("Vehicle speed",                "Hastighet",       "km/h", "#1565C0"),
        ("Engine RPM",                   "Motorturtall",    "rpm",  "#B71C1C"),
        ("Calculated engine load value", "Motorlast",       "%",    "#E65100"),
        ("Calculated instant fuel rate", "Drivstofforbruk", "L/h", "#1B5E20"),
    ]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[p[1] for p in paneler],
        vertical_spacing=0.12, horizontal_spacing=0.08)
    posisjoner = [(1,1),(1,2),(2,1),(2,2)]
    for (pid, navn, enhet, farge), (row, col) in zip(paneler, posisjoner):
        serie = hent_signal(df, pid)
        if serie is None:
            continue
        snitt = serie.dropna().mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=serie, name=navn,
            line=dict(color=farge, width=1.5), fill="tozeroy",
            hovertemplate=f"<b>{navn}</b><br>%{{y:.2f}} {enhet}<extra></extra>",
        ), row=row, col=col)
        fig.add_hline(y=snitt, line_dash="dot", line_color=farge,
            line_width=1, opacity=0.6,
            annotation_text=f"snitt {snitt:.1f}",
            annotation_font_size=9, row=row, col=col)
        fig.update_yaxes(title_text=enhet, row=row, col=col, title_font_size=10)
    fig.update_layout(height=580, showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40), hovermode="x unified")
    return fig


def lag_scatter(df: pd.DataFrame, x_pid: str, y_pid: str, farge_pid) -> go.Figure:
    x_serie = hent_signal(df, x_pid)
    y_serie = hent_signal(df, y_pid)
    if x_serie is None or y_serie is None:
        return go.Figure()
    x_navn  = SIGNALER.get(x_pid, (x_pid, "", ""))[0]
    y_navn  = SIGNALER.get(y_pid, (y_pid, "", ""))[0]
    x_enhet = SIGNALER.get(x_pid, ("", "", ""))[1]
    y_enhet = SIGNALER.get(y_pid, ("", "", ""))[1]
    felles  = pd.DataFrame({"x": x_serie, "y": y_serie})
    if farge_pid and farge_pid != "(ingen)":
        f_serie = hent_signal(df, farge_pid)
        f_navn  = SIGNALER.get(farge_pid, (farge_pid, "", ""))[0]
        f_enhet = SIGNALER.get(farge_pid, ("", "", ""))[1]
        felles["farge"] = f_serie
        felles = felles.dropna()
        fig = px.scatter(felles, x="x", y="y", color="farge",
            color_continuous_scale="YlOrRd",
            labels={"x": f"{x_navn} ({x_enhet})", "y": f"{y_navn} ({y_enhet})",
                    "farge": f"{f_navn} ({f_enhet})"})
    else:
        felles = felles.dropna()
        fig = px.scatter(felles, x="x", y="y",
            labels={"x": f"{x_navn} ({x_enhet})", "y": f"{y_navn} ({y_enhet})"})
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=450, margin=dict(l=60, r=20, t=30, b=50))
    return fig


def lag_histogram(df: pd.DataFrame, pid: str) -> go.Figure:
    serie = hent_signal(df, pid)
    if serie is None:
        return go.Figure()
    navn, enhet, farge = SIGNALER.get(pid, (pid, "", "#607D8B"))
    data = serie.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=40,
        marker_color=farge, opacity=0.75, name=navn))
    fig.update_layout(
        xaxis_title=f"{navn} ({enhet})", yaxis_title="Antall målinger",
        height=350, margin=dict(l=60, r=20, t=20, b=50), showlegend=False)
    return fig

# -----------------------------------------------------------------------------
# STATISTIKK-SAMMENDRAG
# -----------------------------------------------------------------------------

def beregn_statistikk(df: pd.DataFrame) -> dict:
    stats = {}
    stats["varighet_s"]   = df.index.max()
    stats["varighet_min"] = df.index.max() / 60
    s = hent_signal(df, "Vehicle speed")
    if s is not None:
        over_null = s[s > 1]
        stats["toppfart"]    = s.max()
        stats["snittfart"]   = over_null.mean() if len(over_null) > 0 else 0.0
        stats["tomgang_pst"] = (s <= 1).mean() * 100
    s = hent_signal(df, "Engine RPM")
    if s is not None:
        stats["snitt_rpm"] = s.mean()
        stats["maks_rpm"]  = s.max()
    s = hent_signal(df, "Calculated engine load value")
    if s is not None:
        stats["snitt_last"] = s.mean()
        stats["maks_last"]  = s.max()
    s = hent_signal(df, "Calculated instant fuel rate")
    if s is not None:
        stats["snitt_forbruk"] = s.mean()
        stats["maks_forbruk"]  = s.max()
    if "Fuel used" in df.columns:
        v = df["Fuel used"].dropna()
        if len(v) > 0:
            stats["total_drivstoff"] = v.max()
    if "Distance travelled" in df.columns:
        v = df["Distance travelled"].dropna()
        if len(v) > 0:
            stats["distanse"] = v.max()
    return stats

# -----------------------------------------------------------------------------
# BILHELSE-ANALYSE
# -----------------------------------------------------------------------------

def score_farge(score: float) -> str:
    if score >= 8:   return "#2E7D32"
    elif score >= 6: return "#F9A825"
    elif score >= 4: return "#E65100"
    else:            return "#B71C1C"

def score_emoji(score: float) -> str:
    if score >= 8:   return "✅"
    elif score >= 6: return "🟡"
    elif score >= 4: return "🟠"
    else:            return "🔴"


def _gauge_figur(score, tittel, hoyde, font_size, domain_y0, margin_t, margin_b):
    """Intern hjelpefunksjon – bygg én gauge-figur med eksplisitte parametere."""
    farge = score_farge(score)
    # Plotly plasserer "number" i rommet UNDER domain.y[0].
    # Ved å sette domain_y0 kalibrerer vi nøyaktig hvor tall-rommet er,
    # slik at scoren havner sentrert mellom bue-endepunktene.
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [domain_y0, 1.0]},
        number={
            "font": {"size": font_size, "color": farge, "family": "Arial Black"},
            "suffix": "/10",
        },
        title={"text": tittel, "font": {"size": 11}, "align": "center"},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1, "tickfont": {"size": 9}},
            "bar":  {"color": farge, "thickness": 0.3},
            "bgcolor": "rgba(200,200,200,0.15)", "borderwidth": 0,
            "steps": [
                {"range": [0,  4],  "color": "#ffcdd2"},
                {"range": [4,  6],  "color": "#ffe0b2"},
                {"range": [6,  8],  "color": "#fff9c4"},
                {"range": [8, 10],  "color": "#c8e6c9"},
            ],
            "threshold": {"line": {"color": farge, "width": 4},
                          "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(
        height=hoyde,
        margin=dict(l=20, r=20, t=margin_t, b=margin_b),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def lag_gauge(score: float, tittel: str, hoyde: int = 220) -> go.Figure:
    """
    Liten gauge for hver enkelt faktor (brukt inne i expander-kortene).
    Figur: 180px høy. domain_y0=0.18 gir tall sentrert under buen.
    """
    return _gauge_figur(
        score=score, tittel=tittel,
        hoyde=180,
        font_size=30,
        domain_y0=0.22,   # buen starter 22% opp → tall i nedre 22% = linje med endepunkter
        margin_t=10, margin_b=5,
    )


def lag_gauge_total(score: float, tittel: str) -> go.Figure:
    """
    Stor gauge for totalvurdering øverst på bilhelse-siden.
    Figur: 280px høy. domain_y0=0.22 gir bedre proporsjon for større figur.
    """
    return _gauge_figur(
        score=score, tittel=tittel,
        hoyde=280,
        font_size=46,
        domain_y0=0.22,   # større figur → litt mer rom under buen
        margin_t=30, margin_b=10,
    )


def analyser_kaldstart(df: pd.DataFrame) -> dict:
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": "", "tid_serie": None}
    kjoyl = hent_signal(df, "Engine coolant temperature")
    rpm   = hent_signal(df, "Engine RPM")
    if kjoyl is None or rpm is None:
        r["forklaring"] = "Mangler kjølevæsketemperatur eller RPM-data i denne loggen."
        return r
    r["tilgjengelig"] = True
    felles = pd.DataFrame({"temp": kjoyl, "rpm": rpm}).dropna()
    kald = felles[felles["temp"] < 70]
    maks_temp      = felles["temp"].max()
    snitt_rpm_kald = kald["rpm"].mean() if len(kald) > 0 else None
    maks_rpm_kald  = kald["rpm"].max()  if len(kald) > 0 else None
    r["maks_temp"]      = maks_temp
    r["snitt_rpm_kald"] = snitt_rpm_kald
    r["maks_rpm_kald"]  = maks_rpm_kald
    r["tid_serie"]      = felles
    d = []
    if snitt_rpm_kald is None:
        score = 7.0
        d.append("Ikke nok data fra kaldkjøringsfasen til full vurdering.")
    elif snitt_rpm_kald < 1000:
        score = 9.5
        d.append(f"Utmerket: snitt {snitt_rpm_kald:.0f} rpm under kaldkjøring.")
    elif snitt_rpm_kald < 1500:
        score = 7.5
        d.append(f"Bra: snitt {snitt_rpm_kald:.0f} rpm under kaldkjøring.")
    elif snitt_rpm_kald < 2500:
        score = 5.0
        d.append(f"Moderat: {snitt_rpm_kald:.0f} rpm — noe høyt for kald motor.")
    else:
        score = 2.5
        d.append(f"Høy belastning på kald motor: {snitt_rpm_kald:.0f} rpm i snitt.")
    if maks_rpm_kald and maks_rpm_kald > 3000:
        score = max(1.0, score - 2.0)
        d.append(f"Advarsel: {maks_rpm_kald:.0f} rpm maks mens motoren var under 70°C.")
    if maks_temp < 70:
        d.append(f"Merk: kjølevæsken nådde bare {maks_temp:.0f}°C — motoren ble ikke fullt varm i denne loggøkten.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "En kald motor tåler dårligere høy belastning fordi motoroljen ikke har nådd "
        "optimal viskositet og metalldelene ikke er varmeekspandert til riktige toleranser. "
        "Ideelt bør RPM holdes under 1500 de første minuttene etter kaldstart."
    )
    return r


def analyser_drivstoff(df: pd.DataFrame) -> dict:
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": "",
         "snitt": 0, "maks": 0, "forventet_min": 0.6, "forventet_max": 1.2}
    forbruk = hent_signal(df, "Calculated instant fuel rate")
    if forbruk is None:
        r["forklaring"] = "Mangler drivstofforbruk-data."
        return r
    r["tilgjengelig"] = True
    snitt = forbruk.mean()
    maks  = forbruk.max()
    r["snitt"] = snitt
    r["maks"]  = maks
    FMIN, FMAX = 0.6, 1.2
    d = [
        f"Snitt: {snitt:.2f} L/h  |  Maks: {maks:.2f} L/h",
        f"Forventet tomgangsforbruk: {FMIN}–{FMAX} L/h",
    ]
    if snitt <= FMAX:
        score = 9.0
        d.append("Forbruket er innenfor forventet nivå for tomgang.")
    else:
        avvik = ((snitt - FMAX) / FMAX) * 100
        score = max(1.0, 9.0 - (avvik / 10))
        d.append(f"Forbruket er {avvik:.0f}% over forventet tomgangsnivå.")
    maf = hent_signal(df, "MAF air flow rate")
    if maf is not None:
        snitt_maf = maf.mean()
        d.append(f"Snitt luftmasserate (MAF): {snitt_maf:.1f} g/s")
        if snitt_maf > 15:
            score = max(1.0, score - 1.5)
            d.append("Høy MAF på tomgang — mulig luftlekkasje eller sensor-feil.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "Drivstofforbruk på tomgang gjenspeiler motorens grunnleggende effektivitet. "
        "Høyere forbruk enn forventet kan indikere feil i innsprøytning, "
        "slitt tenning, tilstoppet luftfilter, eller karbonoppbygging."
    )
    return r


def analyser_batteri(df: pd.DataFrame) -> dict:
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": "", "snitt_v": 0}
    batt = hent_signal(df, "OBD Module Voltage")
    if batt is None or batt.dropna().empty:
        r["forklaring"] = "Mangler batterispenning-data."
        return r
    r["tilgjengelig"] = True
    data = batt.dropna()
    snitt_v = data.mean()
    min_v   = data.min()
    maks_v  = data.max()
    r["snitt_v"] = snitt_v
    d = [
        f"Snitt: {snitt_v:.2f} V  |  Min: {min_v:.2f} V  |  Maks: {maks_v:.2f} V",
        "Normalt med motor i gang: 13.8 – 14.8 V",
    ]
    if 13.8 <= snitt_v <= 14.8:
        score = 9.5
        d.append("Ladespenning er perfekt — generator og batteri ser bra ut.")
    elif 13.5 <= snitt_v < 13.8:
        score = 7.0
        d.append("Litt lav ladespenning — kan tyde på slitt batteri eller generator.")
    elif snitt_v > 14.8:
        score = 6.0
        d.append("Litt høy spenning — mulig overladning, sjekk generatoren.")
    else:
        score = 4.0
        d.append("Lav ladespenning — batteri eller generator bør sjekkes snart.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "Med motoren i gang lader generatoren batteriet. Normal ladespenning er 13.8–14.8V. "
        "Under 13.5V kan indikere svak generator eller batteri. Over 14.8V kan skade batteriet."
    )
    return r


def analyser_feilkoder(df: pd.DataFrame) -> dict:
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": "", "mil_km": 0}
    mil = hent_signal(df, "Distance traveled with MIL on")
    if mil is None:
        r["forklaring"] = "Mangler MIL-data (Check Engine-lampe)."
        return r
    r["tilgjengelig"] = True
    mil_km = mil.dropna().max()
    r["mil_km"] = mil_km
    d = [f"Kjørt med Check Engine-lampe aktiv: {mil_km:.1f} km"]
    if mil_km == 0:
        score = 10.0
        d.append("Ingen aktive feilkoder registrert.")
    elif mil_km < 10:
        score = 6.0
        d.append("Noe distanse kjørt med MIL på — les ut feilkoder snart.")
    else:
        score = 3.0
        d.append(f"{mil_km:.0f} km kjørt med MIL på — les feilkoder umiddelbart.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "MIL (Malfunction Indicator Lamp) er Check Engine-lampen. "
        "Distansen logget med MIL aktiv viser om bilen har kjørt med en registrert feil. "
        "Selv om lampen ikke lyser nå kan historiske feil vises her."
    )
    return r


def analyser_motorlast(df: pd.DataFrame) -> dict:
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": ""}
    last = hent_signal(df, "Calculated engine load value")
    if last is None:
        r["forklaring"] = "Mangler motorlast-data."
        return r
    r["tilgjengelig"] = True
    snitt = last.mean()
    maks  = last.max()
    d = [
        f"Snitt motorlast: {snitt:.1f}%  |  Maks: {maks:.1f}%",
        "Forventet på tomgang: 20–45%",
    ]
    if 20 <= snitt <= 45:
        score = 9.0
        d.append("Motorlasten er normal for tomgang.")
    elif snitt < 20:
        score = 6.5
        d.append("Lav last — kan være normalt for en liten motor.")
    elif 45 < snitt <= 60:
        score = 6.0
        d.append("Noe høy last på tomgang — mulig karbonoppbygging.")
    else:
        score = 3.5
        d.append("Høy last på tomgang — bør undersøkes av verksted.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "Motorlast på tomgang reflekterer hvor hardt motoren jobber bare for å holde seg i gang. "
        "Høy tomgangslast kan indikere mekanisk motstand, karbonoppbygging på innsugventiler, "
        "slitt tenning, eller at ekstrautstyr (AC, varmepumpe) trekker mye strøm."
    )
    return r


def beregn_total(analyser: dict) -> float:
    vekter = {"kaldstart": 3.0, "drivstoff": 2.0, "batteri": 1.5, "feilkoder": 2.5, "motorlast": 1.0}
    vs, vt = 0.0, 0.0
    for navn, vekt in vekter.items():
        a = analyser.get(navn, {})
        if a.get("score") is not None:
            vs += a["score"] * vekt
            vt += vekt
    return round(vs / vt, 1) if vt > 0 else 0.0


def lag_forbruk_plot(r: dict) -> go.Figure:
    snitt    = r.get("snitt", 0)
    fmin     = r.get("forventet_min", 0.6)
    fmax     = r.get("forventet_max", 1.2)
    fmidt    = (fmin + fmax) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Forventet (midt)", "Faktisk snitt"],
        y=[fmidt, snitt],
        marker_color=["#43A047", score_farge(r.get("score", 5))],
        text=[f"{fmidt:.2f} L/h", f"{snitt:.2f} L/h"],
        textposition="outside", width=0.4,
    ))
    fig.add_hrect(y0=fmin, y1=fmax, fillcolor="#c8e6c9", opacity=0.4,
        line_width=0, annotation_text="Forventet område",
        annotation_position="top right", annotation_font_size=10)
    fig.update_layout(
        yaxis_title="L/h",
        yaxis=dict(range=[0, max(snitt, fmax) * 1.4]),
        height=280, margin=dict(l=40, r=20, t=20, b=40), showlegend=False)
    return fig


def lag_kaldstart_plot(r: dict) -> go.Figure:
    felles = r.get("tid_serie")
    if felles is None or len(felles) == 0:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=felles.index, y=felles["temp"], name="Kjølevæske (°C)",
        line=dict(color="#880E4F", width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=felles.index, y=felles["rpm"], name="RPM",
        line=dict(color="#B71C1C", width=2, dash="dot")), secondary_y=True)
    fig.add_hline(y=70, line_dash="dash", line_color="#43A047", line_width=2,
        annotation_text="70°C — motor varm", annotation_font_size=10,
        secondary_y=False)
    fig.update_yaxes(title_text="Kjølevæske (°C)", secondary_y=False, title_font_color="#880E4F")
    fig.update_yaxes(title_text="RPM", secondary_y=True, title_font_color="#B71C1C")
    fig.update_xaxes(title_text="Tid (sekunder)")
    fig.update_layout(height=300, margin=dict(l=60, r=60, t=20, b=50),
        legend=dict(orientation="h", y=1.1), hovermode="x unified")
    return fig


# -----------------------------------------------------------------------------
# HOVEDAPP
# -----------------------------------------------------------------------------

def main():
    st.title("🚗 OBD2 Kjøredata Analyse")
    st.caption("Car Scanner ELM OBD2 — interaktiv dataanalyse")

    with st.sidebar:
        st.header("Last opp data")
        opplastet_fil = st.file_uploader(
            "Velg CSV-fil fra Car Scanner",
            type=["csv"],
            help="Eksporter fra Car Scanner: Logger → Del som CSV",
        )
        st.divider()
        st.markdown("**Om appen**")
        st.markdown(
            "Laster inn OBD2-data og viser interaktive grafer og helseanalyse. "
            "Zoom, pan og hover direkte i grafene."
        )

    if opplastet_fil is None:
        st.info("Last opp en CSV-fil i sidepanelet for å komme i gang.")
        st.markdown("""
        **Slik eksporterer du fra Car Scanner:**
        1. Åpne Car Scanner på telefonen
        2. Gå til **Logger** (nedre meny)
        3. Velg en kjøreøkt
        4. Trykk **Del** og velg **Eksporter som CSV**
        5. Send filen til PC-en og last den opp her
        """)
        return

    try:
        df = importer_car_scanner(opplastet_fil.read())
    except Exception as e:
        st.error(f"Kunne ikke lese filen: {e}")
        return

    tilgjengelige_pid = [p for p in df.columns if p not in SKJUL]

    pid_til_navn = {
        pid: f"{SIGNALER[pid][0]} ({SIGNALER[pid][1]})" if pid in SIGNALER else pid
        for pid in tilgjengelige_pid
    }
    navn_til_pid = {v: k for k, v in pid_til_navn.items()}

    # ── FANER ──────────────────────────────────────────────────────────────
    fane1, fane2, fane3, fane4, fane5 = st.tabs([
        "📊 Dashboard", "❤️ Bilhelse", "📈 Tidsserier", "🔍 Utforsk", "📋 Rådata"
    ])

    # ── DASHBOARD ──────────────────────────────────────────────────────────
    with fane1:
        stats = beregn_statistikk(df)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Varighet",         f"{stats.get('varighet_min', 0):.1f} min")
        with col2: st.metric("Toppfart",          f"{stats.get('toppfart', 0):.0f} km/h")
        with col3: st.metric("Snitt RPM",         f"{stats.get('snitt_rpm', 0):.0f}")
        with col4: st.metric("Snitt forbruk",     f"{stats.get('snitt_forbruk', 0):.2f} L/h")
        col5, col6, col7, col8 = st.columns(4)
        with col5: st.metric("Snittfart (kjøring)", f"{stats.get('snittfart', 0):.1f} km/h")
        with col6: st.metric("Maks motorlast",    f"{stats.get('maks_last', 0):.1f} %")
        with col7: st.metric("Drivstoff brukt",   f"{stats.get('total_drivstoff', 0):.4f} L")
        with col8: st.metric("Distanse",          f"{stats.get('distanse', 0):.2f} km")
        st.divider()
        st.subheader("Oversikt – fire hoveddisignaler")
        st.plotly_chart(lag_dashboard(df), use_container_width=True)

    # ── BILHELSE ───────────────────────────────────────────────────────────
    with fane2:
        st.subheader("Bilhelse-analyse")
        st.caption(
            "Basert på OBD2-data fra denne kjøreøkten. "
            "Scores er estimater — ikke erstatning for verkstedssjekk."
        )

        # Kjør alle analyser
        analyser = {
            "kaldstart":  analyser_kaldstart(df),
            "drivstoff":  analyser_drivstoff(df),
            "batteri":    analyser_batteri(df),
            "feilkoder":  analyser_feilkoder(df),
            "motorlast":  analyser_motorlast(df),
        }
        total = beregn_total(analyser)

        # ── Totalvurdering ──
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.plotly_chart(lag_gauge_total(total, "Totalvurdering – Bilhelse"),
                            use_container_width=True)
        with col_t2:
            st.markdown("### Oppsummering")
            labels = {
                "kaldstart": "Kaldstart-adferd",
                "drivstoff": "Drivstofforbruk",
                "batteri":   "Batteri / Generator",
                "feilkoder": "Feilkoder (MIL)",
                "motorlast": "Motorlast tomgang",
            }
            for navn, etikett in labels.items():
                a = analyser[navn]
                if a["score"] is not None:
                    s = a["score"]
                    st.markdown(
                        f"{score_emoji(s)} **{etikett}** &nbsp;&nbsp; "
                        f"<span style='color:{score_farge(s)};font-size:1.1em;font-weight:bold'>"
                        f"{s}/10</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"⚪ **{etikett}** — ikke nok data")

        st.divider()

        # ── Detaljkort per faktor ──
        faktorer = [
            ("kaldstart",  "🌡️ Kaldstart-adferd"),
            ("drivstoff",  "⛽ Drivstofforbruk"),
            ("batteri",    "🔋 Batteri / Generator"),
            ("feilkoder",  "🔧 Feilkoder (MIL)"),
            ("motorlast",  "⚙️ Motorlast tomgang"),
        ]

        for navn, tittel in faktorer:
            a = analyser[navn]
            with st.expander(
                f"{tittel}  {'— ' + str(a['score']) + '/10' if a['score'] is not None else '— ingen data'}",
                expanded=False,
            ):
                if not a["tilgjengelig"]:
                    st.warning(a["forklaring"])
                    continue

                col_g, col_d = st.columns([1, 2])

                with col_g:
                    st.plotly_chart(
                        lag_gauge(a["score"], tittel.split(" ", 1)[1]),
                        use_container_width=True,
                    )


                with col_d:
                    st.markdown("**Funn:**")
                    for linje in a["detaljer"]:
                        st.markdown(f"- {linje}")
                    st.info(a["forklaring"])

                # Ekstra plott per faktor
                if navn == "kaldstart" and a.get("tid_serie") is not None:
                    st.markdown("**Kjølevæsketemperatur og RPM over tid:**")
                    st.plotly_chart(lag_kaldstart_plot(a), use_container_width=True)

                if navn == "drivstoff":
                    st.markdown("**Faktisk vs forventet drivstofforbruk:**")
                    st.plotly_chart(lag_forbruk_plot(a), use_container_width=True)

        # ── GPS-info ──
        st.divider()
        st.markdown("### 🗺️ GPS og kjørerute")
        st.info(
            "Denne CSV-filen inneholder ikke GPS-koordinater. "
            "Car Scanner logger ikke GPS automatisk. "
            "For å få kartvisning av kjøreruten, gå til **Innstillinger → Logger** "
            "i Car Scanner og aktiver **GPS-logging** før neste kjøretur. "
            "Da vil CSV-filen inneholde Latitude/Longitude-kolonner som kan vises på kart her."
        )

    # ── TIDSSERIER ─────────────────────────────────────────────────────────
    with fane3:
        st.subheader("Velg signaler å vise")
        standard_valg = [
            pid_til_navn[p] for p in
            ["Vehicle speed", "Engine RPM", "Calculated engine load value",
             "Calculated instant fuel rate", "Vehicle acceleration"]
            if p in pid_til_navn
        ]
        valgte_navn = st.multiselect(
            "Signaler", options=list(pid_til_navn.values()),
            default=standard_valg,
        )
        valgte_pid = [navn_til_pid[n] for n in valgte_navn]
        if valgte_pid:
            st.plotly_chart(lag_tidsserie(df, valgte_pid), use_container_width=True)
        else:
            st.warning("Velg minst ett signal.")

    # ── UTFORSK ────────────────────────────────────────────────────────────
    with fane4:
        st.subheader("Scatter-plott")
        alle_navn = list(pid_til_navn.values())
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            x_valg = st.selectbox("X-akse", alle_navn,
                index=alle_navn.index(pid_til_navn["Engine RPM"]) if "Engine RPM" in pid_til_navn else 0)
        with col_b:
            y_valg = st.selectbox("Y-akse", alle_navn,
                index=alle_navn.index(pid_til_navn["Calculated engine load value"]) if "Calculated engine load value" in pid_til_navn else 1)
        with col_c:
            farge_valg = st.selectbox("Fargekode etter", ["(ingen)"] + alle_navn,
                index=alle_navn.index(pid_til_navn["Calculated instant fuel rate"]) + 1 if "Calculated instant fuel rate" in pid_til_navn else 0)
        x_pid = navn_til_pid[x_valg]
        y_pid = navn_til_pid[y_valg]
        f_pid = navn_til_pid.get(farge_valg) if farge_valg != "(ingen)" else None
        st.plotly_chart(lag_scatter(df, x_pid, y_pid, f_pid), use_container_width=True)
        st.divider()
        st.subheader("Fordeling")
        hist_valg = st.selectbox("Velg signal", alle_navn, key="hist")
        st.plotly_chart(lag_histogram(df, navn_til_pid[hist_valg]), use_container_width=True)

    # ── RÅDATA ─────────────────────────────────────────────────────────────
    with fane5:
        st.subheader("Rådata (bred tabell)")
        st.caption(f"{len(df)} tidspunkter, {len(df.columns)} signaler")
        vis_kolonner = st.multiselect(
            "Velg kolonner å vise", options=list(df.columns),
            default=[p for p in ["Vehicle speed","Engine RPM",
                "Calculated engine load value","Calculated instant fuel rate"]
                if p in df.columns],
        )
        if vis_kolonner:
            st.dataframe(df[vis_kolonner].round(3), use_container_width=True, height=400)
        st.download_button(
            label="Last ned som CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="obd2_bred_tabell.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()