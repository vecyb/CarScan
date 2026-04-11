# -*- coding: utf-8 -*-
"""
OBD2 Analyse - Car Scanner ELM OBD2
=====================================
Streamlit-app med Supabase-integrasjon for lagring av kjørehistorikk.

Kjoer med:
    streamlit run obd2_app.py

Avhengigheter:
    pip install streamlit plotly pandas supabase
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from supabase import create_client
from datetime import datetime

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
# SUPABASE-TILKOBLING
# -----------------------------------------------------------------------------

@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def lagre_kjoretur(rad: dict) -> bool:
    """Lagrer en kjøretur til Supabase. Returnerer True hvis vellykket."""
    try:
        sb = get_supabase()
        sb.table("kjøreturer").insert(rad).execute()
        return True
    except Exception as e:
        st.error(f"Kunne ikke lagre kjøretur: {e}")
        return False


@st.cache_data(ttl=60)
def hent_historikk() -> pd.DataFrame:
    """Henter alle lagrede kjøreturer fra Supabase. Cache i 60 sekunder."""
    try:
        sb = get_supabase()
        res = sb.table("kjøreturer").select("*").order("opprettet", desc=True).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df["opprettet"] = pd.to_datetime(df["opprettet"])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Kunne ikke hente historikk: {e}")
        return pd.DataFrame()


def slett_kjøreturer(ider: list) -> bool:
    """Sletter kjøreturer med gitte UUID-er fra Supabase."""
    try:
        sb = get_supabase()
        for _id in ider:
            sb.table("kjøreturer").delete().eq("id", _id).execute()
        hent_historikk.clear()
        return True
    except Exception as e:
        st.error(f"Kunne ikke slette: {e}")
        return False


def er_duplikat(filnavn: str, varighet_s: float) -> bool:
    """Sjekker om en kjøretur med samme filnavn og varighet allerede er lagret."""
    try:
        sb = get_supabase()
        res = (sb.table("kjøreturer")
               .select("id")
               .eq("filnavn", filnavn)
               .gte("varighet_s", varighet_s - 1.0)
               .lte("varighet_s", varighet_s + 1.0)
               .execute())
        return len(res.data) > 0
    except Exception:
        return False


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
    "Throttle position":                ("Gasspådrag",          "%",     "#F57F17"),
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
    farge = score_farge(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [domain_y0, 1.0]},
        number={"font": {"size": font_size, "color": farge, "family": "Arial Black"}, "suffix": "/10"},
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
            "threshold": {"line": {"color": farge, "width": 4}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(height=hoyde, margin=dict(l=20, r=20, t=margin_t, b=margin_b),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def lag_gauge(score, tittel):
    return _gauge_figur(score, tittel, hoyde=180, font_size=30,
                        domain_y0=0.18, margin_t=30, margin_b=10)

def lag_gauge_total(score, tittel):
    return _gauge_figur(score, tittel, hoyde=280, font_size=46,
                        domain_y0=0.22, margin_t=30, margin_b=10)


def analyser_kaldstart(df):
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
    r["maks_temp"] = maks_temp
    r["snitt_rpm_kald"] = snitt_rpm_kald
    r["maks_rpm_kald"]  = maks_rpm_kald
    r["tid_serie"] = felles
    d = []
    if snitt_rpm_kald is None:
        score = 7.0
        d.append("Ikke nok data fra kaldkjøringsfasen.")
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
        d.append(f"Høy belastning på kald motor: {snitt_rpm_kald:.0f} rpm.")
    if maks_rpm_kald and maks_rpm_kald > 3000:
        score = max(1.0, score - 2.0)
        d.append(f"Advarsel: {maks_rpm_kald:.0f} rpm maks mens motoren var under 70°C.")
    if maks_temp < 70:
        d.append(f"Merk: kjølevæsken nådde bare {maks_temp:.0f}°C i denne loggøkten.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "En kald motor tåler dårligere høy belastning fordi motoroljen ikke har nådd "
        "optimal viskositet. Ideelt bør RPM holdes under 1500 de første minuttene etter kaldstart."
    )
    return r


def analyser_drivstoff(df):
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
    d = [f"Snitt: {snitt:.2f} L/h  |  Maks: {maks:.2f} L/h",
         f"Forventet tomgangsforbruk: {FMIN}–{FMAX} L/h"]
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
            d.append("Høy MAF på tomgang — mulig luftlekkasje.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = (
        "Drivstofforbruk på tomgang gjenspeiler motorens grunnleggende effektivitet. "
        "Høyere enn forventet kan indikere feil i innsprøytning, slitt tenning eller tilstoppet luftfilter."
    )
    return r


def analyser_batteri(df):
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": "", "snitt_v": 0}
    batt = hent_signal(df, "OBD Module Voltage")
    if batt is None or batt.dropna().empty:
        r["forklaring"] = "Mangler batterispenning-data."
        return r
    r["tilgjengelig"] = True
    data = batt.dropna()
    snitt_v = data.mean()
    r["snitt_v"] = snitt_v
    d = [f"Snitt: {snitt_v:.2f} V  |  Min: {data.min():.2f} V  |  Maks: {data.max():.2f} V",
         "Normalt med motor i gang: 13.8 – 14.8 V"]
    if 13.8 <= snitt_v <= 14.8:
        score = 9.5
        d.append("Ladespenning er perfekt.")
    elif 13.5 <= snitt_v < 13.8:
        score = 7.0
        d.append("Litt lav ladespenning — sjekk batteri/generator.")
    elif snitt_v > 14.8:
        score = 6.0
        d.append("Litt høy spenning — mulig overladning.")
    else:
        score = 4.0
        d.append("Lav ladespenning — bør sjekkes snart.")
    r["score"] = round(score, 1)
    r["detaljer"] = d
    r["forklaring"] = "Normal ladespenning med motor i gang er 13.8–14.8V. Avvik kan indikere svak generator eller batteri."
    return r


def analyser_feilkoder(df):
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": "", "mil_km": 0}
    mil = hent_signal(df, "Distance traveled with MIL on")
    if mil is None:
        r["forklaring"] = "Mangler MIL-data."
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
    r["forklaring"] = "MIL (Check Engine-lampen) indikerer registrerte feil. Selv historiske feil vises her."
    return r


def analyser_motorlast(df):
    r = {"tilgjengelig": False, "score": None, "detaljer": [], "forklaring": ""}
    last = hent_signal(df, "Calculated engine load value")
    if last is None:
        r["forklaring"] = "Mangler motorlast-data."
        return r
    r["tilgjengelig"] = True
    snitt = last.mean()
    maks  = last.max()
    d = [f"Snitt motorlast: {snitt:.1f}%  |  Maks: {maks:.1f}%", "Forventet på tomgang: 20–45%"]
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
    r["forklaring"] = "Høy tomgangslast kan indikere karbonoppbygging, slitt tenning eller mekanisk motstand."
    return r


def beregn_total(analyser):
    vekter = {"kaldstart": 3.0, "drivstoff": 2.0, "batteri": 1.5, "feilkoder": 2.5, "motorlast": 1.0}
    vs, vt = 0.0, 0.0
    for navn, vekt in vekter.items():
        a = analyser.get(navn, {})
        if a.get("score") is not None:
            vs += a["score"] * vekt
            vt += vekt
    return round(vs / vt, 1) if vt > 0 else 0.0


def lag_forbruk_plot(r):
    snitt = r.get("snitt", 0)
    fmin, fmax = r.get("forventet_min", 0.6), r.get("forventet_max", 1.2)
    fmidt = (fmin + fmax) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Forventet (midt)", "Faktisk snitt"],
        y=[fmidt, snitt],
        marker_color=["#43A047", score_farge(r.get("score", 5))],
        text=[f"{fmidt:.2f} L/h", f"{snitt:.2f} L/h"],
        textposition="outside", width=0.4,
    ))
    fig.add_hrect(y0=fmin, y1=fmax, fillcolor="#c8e6c9", opacity=0.4, line_width=0,
        annotation_text="Forventet område", annotation_position="top right", annotation_font_size=10)
    fig.update_layout(yaxis_title="L/h", yaxis=dict(range=[0, max(snitt, fmax) * 1.4]),
        height=280, margin=dict(l=40, r=20, t=20, b=40), showlegend=False)
    return fig


def lag_kaldstart_plot(r):
    felles = r.get("tid_serie")
    if felles is None or len(felles) == 0:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=felles.index, y=felles["temp"], name="Kjølevæske (°C)",
        line=dict(color="#880E4F", width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=felles.index, y=felles["rpm"], name="RPM",
        line=dict(color="#B71C1C", width=2, dash="dot")), secondary_y=True)
    fig.add_hline(y=70, line_dash="dash", line_color="#43A047", line_width=2,
        annotation_text="70°C — motor varm", annotation_font_size=10, secondary_y=False)
    fig.update_yaxes(title_text="Kjølevæske (°C)", secondary_y=False, title_font_color="#880E4F")
    fig.update_yaxes(title_text="RPM", secondary_y=True, title_font_color="#B71C1C")
    fig.update_xaxes(title_text="Tid (sekunder)")
    fig.update_layout(height=300, margin=dict(l=60, r=60, t=20, b=50),
        legend=dict(orientation="h", y=1.1), hovermode="x unified")
    return fig

# -----------------------------------------------------------------------------
# HISTORIKK-PLOTT
# -----------------------------------------------------------------------------

def lag_historikk_trend(hist: pd.DataFrame, kolonne: str, tittel: str, farge: str) -> go.Figure:
    """Linjeplott av én metrikk over alle lagrede kjøreturer."""
    data = hist[["opprettet", kolonne]].dropna().sort_values("opprettet")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["opprettet"], y=data[kolonne],
        mode="lines+markers",
        line=dict(color=farge, width=2),
        marker=dict(size=7, color=farge),
        hovertemplate=f"<b>{tittel}</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.2f}}<extra></extra>",
    ))
    # Glidende snitt hvis nok datapunkter
    if len(data) >= 3:
        data["snitt"] = data[kolonne].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=data["opprettet"], y=data["snitt"],
            mode="lines", name="Glidende snitt (3)",
            line=dict(color=farge, width=1.5, dash="dot"),
            opacity=0.6,
        ))
    fig.update_layout(
        title=tittel, height=280,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=len(data) >= 3,
        hovermode="x unified",
        yaxis_title="",
        xaxis_title="",
    )
    return fig


def lag_score_historikk(hist: pd.DataFrame) -> go.Figure:
    """Alle helssescorer over tid i ett plott."""
    score_kol = {
        "total_score":     ("Totalscorecore",    "#1565C0"),
        "kaldstart_score": ("Kaldstart",         "#880E4F"),
        "drivstoff_score": ("Drivstoff",         "#1B5E20"),
        "motorlast_score": ("Motorlast",         "#E65100"),
    }
    fig = go.Figure()
    for kol, (navn, farge) in score_kol.items():
        if kol in hist.columns:
            data = hist[["opprettet", kol]].dropna().sort_values("opprettet")
            if len(data) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=data["opprettet"], y=data[kol],
                mode="lines+markers", name=navn,
                line=dict(color=farge, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{navn}</b>: %{{y:.1f}}/10<extra></extra>",
            ))
    fig.update_layout(
        height=350,
        yaxis=dict(title="Score (1–10)", range=[0, 10.5]),
        xaxis_title="",
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    # Referanselinjer
    fig.add_hline(y=8, line_dash="dash", line_color="#2E7D32",
                  opacity=0.4, annotation_text="Bra (8)", annotation_font_size=9)
    fig.add_hline(y=6, line_dash="dash", line_color="#F9A825",
                  opacity=0.4, annotation_text="OK (6)", annotation_font_size=9)
    return fig


# -----------------------------------------------------------------------------
# HOVEDAPP
# -----------------------------------------------------------------------------

def main():
    st.title("🚗 OBD2 Kjøredata Analyse")
    st.caption("Car Scanner ELM OBD2 — interaktiv dataanalyse med historikk")

    with st.sidebar:
        st.header("Last opp data")
        opplastet_fil = st.file_uploader(
            "Velg CSV-fil fra Car Scanner",
            type=["csv"],
            help="Car Scanner → Logger → Del som CSV",
        )
        st.divider()
        st.markdown("**Om appen**")
        st.markdown(
            "Laster inn OBD2-data, viser interaktive grafer, "
            "helseanalyse og lagrer historikk automatisk."
        )

    # ── INGEN FIL LASTET OPP ───────────────────────────────────────────────
    if opplastet_fil is None:
        # Vis historikkoversikt selv uten fil
        hist = hent_historikk()
        if not hist.empty:
            st.subheader(f"📚 Historikk – {len(hist)} lagrede kjøreturer")
            fane_h1, fane_h2, fane_h3 = st.tabs(["📈 Trender", "🏆 Scorer", "📋 Alle turer"])

            with fane_h1:
                col1, col2 = st.columns(2)
                with col1:
                    if "snitt_forbruk" in hist.columns:
                        st.plotly_chart(lag_historikk_trend(hist, "snitt_forbruk",
                            "Snitt drivstofforbruk (L/h)", "#1B5E20"), use_container_width=True)
                    if "snitt_rpm" in hist.columns:
                        st.plotly_chart(lag_historikk_trend(hist, "snitt_rpm",
                            "Snitt RPM", "#B71C1C"), use_container_width=True)
                with col2:
                    if "snitt_last" in hist.columns:
                        st.plotly_chart(lag_historikk_trend(hist, "snitt_last",
                            "Snitt motorlast (%)", "#E65100"), use_container_width=True)
                    if "maks_temp" in hist.columns:
                        st.plotly_chart(lag_historikk_trend(hist, "maks_temp",
                            "Maks kjølevæsketemp (°C)", "#880E4F"), use_container_width=True)

            with fane_h2:
                st.plotly_chart(lag_score_historikk(hist), use_container_width=True)

            with fane_h3:
                vis_kol = ["opprettet", "filnavn", "varighet_s", "snitt_rpm",
                           "snitt_forbruk", "total_score", "kaldstart_score"]
                vis_kol = [k for k in vis_kol if k in hist.columns]
                st.dataframe(
                    hist[vis_kol].rename(columns={
                        "opprettet": "Dato", "filnavn": "Fil",
                        "varighet_s": "Varighet (s)", "snitt_rpm": "Snitt RPM",
                        "snitt_forbruk": "Forbruk (L/h)", "total_score": "Totalscore",
                        "kaldstart_score": "Kaldstart",
                    }).round(2),
                    use_container_width=True, height=400,
                )
        else:
            st.info("Last opp en CSV-fil i sidepanelet for å komme i gang.")
            st.markdown("""
            **Slik eksporterer du fra Car Scanner:**
            1. Åpne Car Scanner på telefonen
            2. Gå til **Logger** (nedre meny)
            3. Velg en kjøreøkt
            4. Trykk **Del** og velg **Eksporter som CSV**
            5. Last opp filen her
            """)
        return

    # ── LES INN FIL ────────────────────────────────────────────────────────
    try:
        filinnhold = opplastet_fil.read()
        df = importer_car_scanner(filinnhold)
    except Exception as e:
        st.error(f"Kunne ikke lese filen: {e}")
        return

    tilgjengelige_pid = [p for p in df.columns if p not in SKJUL]
    pid_til_navn = {
        pid: f"{SIGNALER[pid][0]} ({SIGNALER[pid][1]})" if pid in SIGNALER else pid
        for pid in tilgjengelige_pid
    }
    navn_til_pid = {v: k for k, v in pid_til_navn.items()}

    # ── ANALYSER (kjøres en gang og deles mellom faner) ────────────────────
    stats   = beregn_statistikk(df)
    analyser = {
        "kaldstart":  analyser_kaldstart(df),
        "drivstoff":  analyser_drivstoff(df),
        "batteri":    analyser_batteri(df),
        "feilkoder":  analyser_feilkoder(df),
        "motorlast":  analyser_motorlast(df),
    }
    total = beregn_total(analyser)

    # ── LAGRE TIL SUPABASE (automatisk, med duplikat-sjekk) ──────────────
    varighet_s_verdi = float(stats.get("varighet_s", 0))
    cache_key = f"lagret_{opplastet_fil.name}_{varighet_s_verdi:.0f}"
    if cache_key not in st.session_state:
        if er_duplikat(opplastet_fil.name, varighet_s_verdi):
            st.session_state[cache_key] = True  # merk som allerede lagret
        else:
            rad = {
                "filnavn":        opplastet_fil.name,
                "opprettet":      datetime.utcnow().isoformat(),
                "varighet_s":     float(stats.get("varighet_s", 0)),
                "snitt_rpm":      float(stats.get("snitt_rpm", 0)) if stats.get("snitt_rpm") else None,
                "maks_rpm":       float(stats.get("maks_rpm", 0))  if stats.get("maks_rpm")  else None,
                "snitt_last":     float(stats.get("snitt_last", 0)) if stats.get("snitt_last") else None,
                "maks_last":      float(stats.get("maks_last", 0))  if stats.get("maks_last")  else None,
                "snitt_forbruk":  float(stats.get("snitt_forbruk", 0)) if stats.get("snitt_forbruk") else None,
                "maks_forbruk":   float(stats.get("maks_forbruk", 0))  if stats.get("maks_forbruk")  else None,
                "total_drivstoff":float(stats.get("total_drivstoff", 0)) if stats.get("total_drivstoff") else None,
                "distanse":       float(stats.get("distanse", 0)) if stats.get("distanse") else None,
                "maks_temp":      float(analyser["kaldstart"].get("maks_temp", 0)) if analyser["kaldstart"].get("maks_temp") else None,
                "snitt_maf":      None,
                "batteri_v":      float(analyser["batteri"].get("snitt_v", 0)) if analyser["batteri"].get("snitt_v") else None,
                "kaldstart_score":analyser["kaldstart"].get("score"),
                "drivstoff_score":analyser["drivstoff"].get("score"),
                "batteri_score":  analyser["batteri"].get("score"),
                "feilkode_score": analyser["feilkoder"].get("score"),
                "motorlast_score":analyser["motorlast"].get("score"),
                "total_score":    float(total),
            }
            if lagre_kjoretur(rad):
                st.session_state[cache_key] = True
                st.toast("Kjøretur lagret!", icon="✅")
                hent_historikk.clear()

    # ── FANER ──────────────────────────────────────────────────────────────
    fane1, fane2, fane3, fane4, fane5, fane6 = st.tabs([
        "📊 Dashboard", "❤️ Bilhelse", "📚 Historikk",
        "📈 Tidsserier", "🔍 Utforsk", "📋 Rådata"
    ])

    # ── DASHBOARD ──────────────────────────────────────────────────────────
    with fane1:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Varighet",           f"{stats.get('varighet_min', 0):.1f} min")
        with col2: st.metric("Toppfart",            f"{stats.get('toppfart', 0):.0f} km/h")
        with col3: st.metric("Snitt RPM",           f"{stats.get('snitt_rpm', 0):.0f}")
        with col4: st.metric("Snitt forbruk",       f"{stats.get('snitt_forbruk', 0):.2f} L/h")
        col5, col6, col7, col8 = st.columns(4)
        with col5: st.metric("Snittfart (kjøring)", f"{stats.get('snittfart', 0):.1f} km/h")
        with col6: st.metric("Maks motorlast",      f"{stats.get('maks_last', 0):.1f} %")
        with col7: st.metric("Drivstoff brukt",     f"{stats.get('total_drivstoff', 0):.4f} L")
        with col8: st.metric("Distanse",            f"{stats.get('distanse', 0):.2f} km")
        st.divider()
        st.subheader("Oversikt – fire hoveddisignaler")
        st.plotly_chart(lag_dashboard(df), use_container_width=True)

    # ── BILHELSE ───────────────────────────────────────────────────────────
    with fane2:
        st.subheader("Bilhelse-analyse")
        st.caption("Scores er estimater — ikke erstatning for verkstedssjekk.")

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
                        f"{score_emoji(s)} **{etikett}** &nbsp;&nbsp;"
                        f"<span style='color:{score_farge(s)};font-size:1.1em;font-weight:bold'>"
                        f"{s}/10</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"⚪ **{etikett}** — ikke nok data")

        st.divider()
        faktorer = [
            ("kaldstart", "🌡️ Kaldstart-adferd"),
            ("drivstoff", "⛽ Drivstofforbruk"),
            ("batteri",   "🔋 Batteri / Generator"),
            ("feilkoder", "🔧 Feilkoder (MIL)"),
            ("motorlast", "⚙️ Motorlast tomgang"),
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
                    st.plotly_chart(lag_gauge(a["score"], tittel.split(" ", 1)[1]),
                                    use_container_width=True)
                with col_d:
                    st.markdown("**Funn:**")
                    for linje in a["detaljer"]:
                        st.markdown(f"- {linje}")
                    st.info(a["forklaring"])
                if navn == "kaldstart" and a.get("tid_serie") is not None:
                    st.markdown("**Kjølevæsketemperatur og RPM over tid:**")
                    st.plotly_chart(lag_kaldstart_plot(a), use_container_width=True)
                if navn == "drivstoff":
                    st.markdown("**Faktisk vs forventet drivstofforbruk:**")
                    st.plotly_chart(lag_forbruk_plot(a), use_container_width=True)

    # ── HISTORIKK ──────────────────────────────────────────────────────────
    with fane3:
        hist = hent_historikk()
        if hist.empty:
            st.info("Ingen historikk enda. Last opp flere kjøreturer for å se trender.")
        else:
            st.subheader(f"📚 {len(hist)} lagrede kjøreturer")

            # ── Velg tur fra historikk for analyse ────────────────────────
            st.markdown("#### Åpne en tidligere tur for analyse")
            hist_visning = hist.copy()
            hist_visning["visningsnavn"] = hist_visning.apply(
                lambda r: f"{r['opprettet'].strftime('%d.%m.%Y %H:%M')}  —  {r['filnavn']}  "
                          f"(score: {r['total_score']:.1f}/10)", axis=1
            )
            valgt_visning = st.selectbox(
                "Velg tur",
                options=["— Vis opplastet fil —"] + list(hist_visning["visningsnavn"]),
                key="historikk_valg",
            )
            if valgt_visning != "— Vis opplastet fil —":
                valgt_rad = hist_visning[hist_visning["visningsnavn"] == valgt_visning].iloc[0]
                st.info(
                    f"Du har valgt **{valgt_rad['filnavn']}** fra {valgt_rad['opprettet'].strftime('%d.%m.%Y %H:%M')}. "
                    f"For full grafanalyse av denne turen må du laste opp CSV-filen på nytt — "
                    f"dataene under viser lagrede nøkkeltall fra da filen ble importert."
                )
                kol1, kol2, kol3, kol4 = st.columns(4)
                with kol1: st.metric("Totalscore", f"{valgt_rad.get('total_score', 0):.1f}/10")
                with kol2: st.metric("Snitt RPM", f"{valgt_rad.get('snitt_rpm', 0):.0f}")
                with kol3: st.metric("Forbruk", f"{valgt_rad.get('snitt_forbruk', 0):.2f} L/h")
                with kol4: st.metric("Varighet", f"{valgt_rad.get('varighet_s', 0)/60:.1f} min")

            st.divider()

            # ── Sammenligning med historisk snitt ─────────────────────────
            if len(hist) > 1:
                st.markdown("#### Denne turen vs historisk snitt")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    hist_snitt = hist["total_score"].mean()
                    st.metric("Totalscore", f"{total:.1f}/10",
                              delta=f"{total - hist_snitt:+.1f} vs snitt")
                with col2:
                    if "snitt_forbruk" in hist.columns:
                        h_snitt = hist["snitt_forbruk"].mean()
                        n_snitt = stats.get("snitt_forbruk", 0)
                        st.metric("Forbruk", f"{n_snitt:.2f} L/h",
                                  delta=f"{n_snitt - h_snitt:+.2f} vs snitt",
                                  delta_color="inverse")
                with col3:
                    if "snitt_rpm" in hist.columns:
                        h_snitt = hist["snitt_rpm"].mean()
                        n_snitt = stats.get("snitt_rpm", 0)
                        st.metric("Snitt RPM", f"{n_snitt:.0f}",
                                  delta=f"{n_snitt - h_snitt:+.0f} vs snitt")
                with col4:
                    if "snitt_last" in hist.columns:
                        h_snitt = hist["snitt_last"].mean()
                        n_snitt = stats.get("snitt_last", 0)
                        st.metric("Motorlast", f"{n_snitt:.1f}%",
                                  delta=f"{n_snitt - h_snitt:+.1f}% vs snitt")
                st.divider()

            # ── Scorer og trender ─────────────────────────────────────────
            st.markdown("#### Helssescorer over tid")
            st.plotly_chart(lag_score_historikk(hist), use_container_width=True)

            st.markdown("#### Nøkkeltrender")
            col1, col2 = st.columns(2)
            with col1:
                if "snitt_forbruk" in hist.columns:
                    st.plotly_chart(lag_historikk_trend(hist, "snitt_forbruk",
                        "Drivstofforbruk (L/h)", "#1B5E20"), use_container_width=True)
                if "snitt_rpm" in hist.columns:
                    st.plotly_chart(lag_historikk_trend(hist, "snitt_rpm",
                        "Snitt RPM", "#B71C1C"), use_container_width=True)
            with col2:
                if "snitt_last" in hist.columns:
                    st.plotly_chart(lag_historikk_trend(hist, "snitt_last",
                        "Motorlast (%)", "#E65100"), use_container_width=True)
                if "maks_temp" in hist.columns:
                    st.plotly_chart(lag_historikk_trend(hist, "maks_temp",
                        "Maks kjølevæsketemp (°C)", "#880E4F"), use_container_width=True)

            # ── Tabell med sletting ───────────────────────────────────────
            st.markdown("#### Alle turer")
            st.caption("Velg turer i tabellen og trykk slett for å fjerne dem.")

            vis_kol = ["id", "opprettet", "filnavn", "varighet_s", "snitt_rpm",
                       "snitt_forbruk", "total_score", "kaldstart_score",
                       "drivstoff_score", "motorlast_score"]
            vis_kol = [k for k in vis_kol if k in hist.columns]

            tabell_data = hist[vis_kol].copy()
            tabell_data["opprettet"] = tabell_data["opprettet"].dt.strftime("%d.%m.%Y %H:%M")

            # Bruk st.data_editor for avkrysningsbokser
            tabell_data.insert(0, "Velg", False)
            redigert = st.data_editor(
                tabell_data.rename(columns={
                    "opprettet": "Dato", "filnavn": "Fil",
                    "varighet_s": "Varighet (s)", "snitt_rpm": "RPM",
                    "snitt_forbruk": "Forbruk (L/h)", "total_score": "Total",
                    "kaldstart_score": "Kaldstart", "drivstoff_score": "Drivstoff",
                    "motorlast_score": "Motorlast",
                }).round(2),
                column_config={
                    "Velg": st.column_config.CheckboxColumn("Velg", default=False),
                    "id": None,  # skjul UUID-kolonnen
                },
                disabled=[c for c in tabell_data.columns if c not in ["Velg", "id"]],
                use_container_width=True,
                height=350,
                key="tur_tabell",
            )

            # Finn valgte rader
            valgte_rader = redigert[redigert["Velg"] == True]
            antall_valgt = len(valgte_rader)

            col_slett1, col_slett2 = st.columns([1, 4])
            with col_slett1:
                slett_knapp = st.button(
                    f"🗑️ Slett {antall_valgt} tur{'er' if antall_valgt != 1 else ''}",
                    disabled=(antall_valgt == 0),
                    type="primary",
                )
            with col_slett2:
                if antall_valgt > 0:
                    st.caption(f"{antall_valgt} tur(er) valgt")

            if slett_knapp and antall_valgt > 0:
                # Hent UUID-er fra original tabell basert på indeks
                valgte_indekser = valgte_rader.index.tolist()
                ider = tabell_data.loc[valgte_indekser, "id"].tolist()
                if slett_kjøreturer(ider):
                    st.success(f"Slettet {antall_valgt} kjøretur(er).")
                    st.rerun()

    # ── TIDSSERIER ─────────────────────────────────────────────────────────
    with fane4:
        st.subheader("Velg signaler å vise")
        standard_valg = [
            pid_til_navn[p] for p in
            ["Vehicle speed", "Engine RPM", "Calculated engine load value",
             "Calculated instant fuel rate", "Vehicle acceleration"]
            if p in pid_til_navn
        ]
        valgte_navn = st.multiselect("Signaler", options=list(pid_til_navn.values()),
                                     default=standard_valg)
        valgte_pid = [navn_til_pid[n] for n in valgte_navn]
        if valgte_pid:
            st.plotly_chart(lag_tidsserie(df, valgte_pid), use_container_width=True)
        else:
            st.warning("Velg minst ett signal.")

    # ── UTFORSK ────────────────────────────────────────────────────────────
    with fane5:
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
    with fane6:
        st.subheader("Rådata (bred tabell)")
        st.caption(f"{len(df)} tidspunkter, {len(df.columns)} signaler")
        vis_kolonner = st.multiselect(
            "Velg kolonner å vise", options=list(df.columns),
            default=[p for p in ["Vehicle speed", "Engine RPM",
                "Calculated engine load value", "Calculated instant fuel rate"]
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
