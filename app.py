from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Load environment variables
load_dotenv(Path("model_datas/azcon_data/.env"))

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder

    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False


st.set_page_config(
    page_title="Persona Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_DIR = Path("model_datas/azcon_data")
SOURCE_FILES = {
    "Aztelekom": "aztelekom.csv",
    "AzAL": "azal.csv",
    "Azerpoct": "azerpoct.csv",
    "ADY": "ady.csv",
    "AzInTelekom": "azintelekom.csv",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(900px circle at 8% 0%, rgba(16, 185, 129, 0.20), transparent 38%),
                    radial-gradient(850px circle at 95% 10%, rgba(59, 130, 246, 0.22), transparent 36%),
                    linear-gradient(180deg, #071224 0%, #0d1b2f 100%);
                color: #e2e8f0;
            }

            h1, h2, h3, h4, .stMarkdown, [data-testid="stMetricLabel"] {
                color: #e2e8f0 !important;
            }

            [data-testid="stMetricValue"] {
                color: #f8fafc;
            }

            .hero {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 20px;
                padding: 1.2rem 1.3rem;
                background: linear-gradient(125deg, rgba(15, 23, 42, 0.72), rgba(12, 30, 60, 0.72));
                box-shadow: 0 15px 40px rgba(2, 6, 23, 0.38);
                margin-bottom: 1rem;
            }

            .chip {
                display: inline-block;
                background: rgba(16, 185, 129, 0.18);
                color: #34d399;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.55rem;
            }

            .panel {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 16px;
                background: linear-gradient(130deg, rgba(15, 23, 42, 0.75), rgba(17, 24, 39, 0.7));
                box-shadow: 0 10px 28px rgba(2, 6, 23, 0.34);
                padding: 0.9rem;
            }

            .small-stat {
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 14px;
                background: rgba(2, 6, 23, 0.35);
                padding: 0.7rem;
                margin-bottom: 0.5rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0a1426 0%, #0c1a31 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.15);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_all_data() -> Dict[str, pd.DataFrame]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    data: Dict[str, pd.DataFrame] = {}
    for company, file_name in SOURCE_FILES.items():
        p = DATA_DIR / file_name
        if p.exists():
            df = pd.read_csv(p)
            df["source_company"] = company
            data[company] = df

    for artifact in ["feature_store.csv", "curated_features.csv", "recommendations.csv"]:
        p = DATA_DIR / artifact
        if p.exists():
            data[p.stem] = pd.read_csv(p)

    if "feature_store" not in data:
        raise FileNotFoundError("feature_store.csv is required under model_datas/azcon_data")

    return data


def build_unified_profiles(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    fs = data["feature_store"].copy()

    if "curated_features" in data:
        curated = data["curated_features"].copy()
        cols = [
            "phone_number",
            "segment",
            "churn_tier",
            "cross_sell_potential",
            "churn_score",
            "revenue_score",
            "digital_score",
            "loyalty_score",
            "mobility_score",
        ]
        cols = [c for c in cols if c in curated.columns]
        curated = curated[cols].drop_duplicates("phone_number")
        fs = fs.merge(curated, on="phone_number", how="left", suffixes=("", "_curated"))

    if "recommendations" in data:
        rec = data["recommendations"].copy()
        rec_summary = (
            rec.groupby("phone_number", as_index=False)
            .agg(
                rec_count=("offer", "count"),
                top_target=("target_company", "first"),
                top_offer=("offer", "first"),
            )
            .fillna({"top_target": "-", "top_offer": "-"})
        )
        fs = fs.merge(rec_summary, on="phone_number", how="left")
    else:
        fs["rec_count"] = 0
        fs["top_target"] = "-"
        fs["top_offer"] = "-"

    numeric_cols = [
        "monthly_spend",
        "total_spent",
        "spend",
        "cltv",
        "avg_parcel_value",
        "revenue_score",
        "flights_per_month",
        "trips",
        "parcels_sent",
        "parcels_received",
        "login_freq",
        "services",
        "sso_usage",
        "backup_usage",
        "loyalty_points",
        "loyalty",
        "churn_score",
        "cross_sell_potential",
    ]
    for col in numeric_cols:
        if col not in fs.columns:
            fs[col] = 0.0
        fs[col] = pd.to_numeric(fs[col], errors="coerce").fillna(0.0)

    fs["total_customer_value"] = (
        fs["monthly_spend"] * 6
        + fs["total_spent"]
        + fs["spend"] * 2
        + fs["cltv"] * 0.45
        + fs["avg_parcel_value"] * 1.4
        + fs["revenue_score"] * 0.8
    ).round(2)

    for col, fallback in {
        "segment": "Unknown",
        "churn_tier": "Unknown",
        "top_target": "-",
        "top_offer": "-",
    }.items():
        if col not in fs.columns:
            fs[col] = fallback
        fs[col] = fs[col].fillna(fallback)

    if "rec_count" not in fs.columns:
        fs["rec_count"] = 0
    fs["rec_count"] = pd.to_numeric(fs["rec_count"], errors="coerce").fillna(0).astype(int)

    fs["travel_activity"] = (fs["flights_per_month"] * 6 + fs["trips"]).round(2)
    fs["digital_activity"] = (
        fs["login_freq"] * 2 + fs["services"] * 8 + fs["sso_usage"] * 12 + fs["backup_usage"] * 5
    ).round(2)
    fs["postal_activity"] = (fs["parcels_sent"] + fs["parcels_received"]).round(2)
    fs["loyalty_activity"] = (fs["loyalty"] + fs["loyalty_points"] / 1500).round(2)

    behavior_score = fs["travel_activity"] * 0.35 + fs["digital_activity"] * 0.45 + fs["postal_activity"] * 0.2
    fs["behavior_persona"] = pd.cut(
        behavior_score,
        bins=[-np.inf, 35, 80, 130, np.inf],
        labels=["Low Activity", "Balanced", "Power User", "Ecosystem Champion"],
    ).astype(str)

    return fs


def available_companies_by_phone(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for company in SOURCE_FILES:
        if company in data:
            rows.append(data[company][["phone_number"]].assign(company=company))
    joined = pd.concat(rows, ignore_index=True)
    return joined.drop_duplicates()


def render_top_nav() -> None:
    st.markdown("""
        <style>
        .top-nav-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e293b;
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
            margin-bottom: 1rem;
            border-radius: 12px;
        }
        .nav-logo {
            font-size: 1.5rem;
            font-weight: 800;
            color: #38bdf8;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .nav-search {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            padding: 0.4rem 1rem;
            color: #f8fafc;
            width: 250px;
        }
        .nav-profile {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 0.4rem 1rem;
            border-radius: 8px;
        }
        </style>
        <div class="top-nav-container">
            <div class="nav-logo">
                Persona
            </div>
            <div style="display:flex; gap:15px; align-items:center;">
                <div class="nav-profile">
                    <span style="color:#f8fafc; font-weight:600;">Sistem İdarəçisi</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def sidebar_filters(df: pd.DataFrame) -> Dict[str, object]:
    with st.sidebar:
        st.header("Ümumi Filtrlər")
        
        segments = st.multiselect("Seqment", sorted(df["segment"].dropna().unique()), default=sorted(df["segment"].dropna().unique()))
        churn_tiers = st.multiselect("Tərketmə Riski", sorted(df["churn_tier"].dropna().unique()), default=sorted(df["churn_tier"].dropna().unique()))
        
        targets = sorted(df["top_target"].dropna().unique())
        target_filter = st.multiselect("Hədəf Şirkət", targets, default=targets)
        
        personas = sorted(df["behavior_persona"].dropna().unique())
        persona_filter = st.multiselect("Davranış Personası", personas, default=personas)
        
        min_val = float(df["total_customer_value"].min())
        max_val = float(df["total_customer_value"].max())
        if min_val < max_val:
            value_range = st.slider("Dəyər Aralığı (Customer Value)", min_val, max_val, (min_val, max_val))
        else:
            value_range = (min_val, max_val)

        max_company = int(max(df.get("company_count", pd.Series([1])).max(), 1))
        company_range = st.slider("Qurum Əhatəsi (Şirkət sayı)", 1, max_company, (1, max_company)) if max_company > 1 else (1, 1)
        
    return {
        "segments": segments,
        "churn_tiers": churn_tiers,
        "targets": target_filter,
        "personas": persona_filter,
        "value_range": value_range,
        "company_range": company_range,
    }


def apply_filters(df: pd.DataFrame, f: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()

    out = out[out["segment"].isin(f["segments"])]
    out = out[out["churn_tier"].isin(f["churn_tiers"])]
    out = out[out["top_target"].isin(f["targets"])]
    out = out[out["behavior_persona"].isin(f["personas"])]
    out = out[(out["total_customer_value"] >= f["value_range"][0]) & (out["total_customer_value"] <= f["value_range"][1])]
    out = out[(out["company_count"] >= f["company_range"][0]) & (out["company_count"] <= f["company_range"][1])]

    return out


def draw_kpis(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    total_customers = len(df)
    
    # Actionable metrics calculation
    at_risk_df = df[df["churn_score"] > 0.7]
    at_risk_value = float(at_risk_df["total_customer_value"].sum()) if not at_risk_df.empty else 0.0
    
    high_xsell = int((df["cross_sell_potential"] >= 0.7).sum()) if len(df) > 0 else 0
    total_value = float(df["total_customer_value"].sum()) if not df.empty else 0.0

    c1.metric("Aktiv Müştəri Bazası", f"{total_customers:,}")
    c2.metric("Ümumi Dəyər (CLTV)", f"${total_value:,.0f}")
    c3.metric("Riskdə Olan Dəyər (!)", f"${at_risk_value:,.0f}", delta="-Təcili Müdaxilə", delta_color="inverse")
    c4.metric("Cross-Sell Mümkünlüyü", f"{high_xsell:,}", delta="+Yüksək Potensial", delta_color="normal")


def generate_ai_recommendations(user: pd.Series) -> list:
    if not HAS_GROQ:
        st.error("Groq paketi tapılmadı. Zəhmət olmasa 'pip install groq python-dotenv' işə salın.")
        return []
    
    api_key = os.environ.get("LLM")
    if not api_key:
        st.error(".env faylında 'LLM' API açarı tapılmadı.")
        return []
        
    client = Groq(api_key=api_key)
    
    prompt = f"""Roldasan: Sən Persona sisteminin məlumat analitiki və marketinq mütəxəssisisən. Şirkət müxtəlif qurumları (Aztelekom, AzAL, Azərpoçt, ADY, AzInTelekom) özündə birləşdirir.
Müştəri məlumatları belədir:
- Nömrə: {user['phone_number']}
- Segment: {user.get('segment', 'Bilinmir')}
- Dəyəri (CLTV): ${user.get('total_customer_value', 0)}
- Tərketmə Riski: {user.get('churn_tier', 'Bilinmir')}
- Davranış Personası: {user.get('behavior_persona', 'Bilinmir')}

Səyahət Aktivliyi: {user.get('travel_activity', 0)} (AzAL/ADY)
Rəqəmsal Aktivlik: {user.get('digital_activity', 0)} (İnternet/Data)
Sadiqlik/Təkrar İstif.: {100 - (user.get('churn_score', 0) * 100)}

Müştərinin bu göstəricilərinə və davranışlarına baxaraq, ona uyğun ən yaxşı 2 və ya 3 ədəd ağıllı çarpaz satış (cross-sell) və ya saxlama (retention) kampaniyası tövsiyə et.
VACİB ŞƏRTLƏR:
1. Qətiyyən emojilərdən istifadə etmə!
2. Tamamilə Azərbaycan dilində yaz.
3. Səbəb (trigger) hissəsində çox birbaşa, qısa və aydın izah ver (məsələn: "Müştəri yüksək dəyərə malikdir, lakin rəqəmsal aktivliyi aşağı düşüb. İnternet paketi təklif edilməlidir").
4. Çox REALİST ol. Qeyri-mümkün və xəyali xidmətlər uydurma. Mövcud qurumların gerçək fəaliyyət sahələrinə aid təkliflər ver: Aztelekom (İnternet yenilənməsi, IPTV, Fiber optik), AzAL (Yeni uçuş istiqaməti üzrə endirim, baqaj, mil hesabı), Azərpoçt (Kuryer xidməti, bağlama endirimi), ADY (Sürətli qatar bileti), AzInTelekom (Bulud saxlac anbarı, SİMA rəqəmsal xidməti).
5. Mütləq JSON formatında qaytar! JSON strukturu belə olmalıdır:
{{
  "recommendations": [
    {{
      "offer": "Kampaniyanın adı/təklif",
      "target_company": "Hədəf qurum (məs. Aztelekom və ya AzAL)",
      "trigger": "Niyə bu təklif uyğundur (birbaşa səbəb)"
    }}
  ]
}}
Başqa heç bir izahat və ya format pozuntusu olmadan yalnız yuxarıdakı quruluşda JSON qaytar.
"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b", 
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None
        )
        content = completion.choices[0].message.content
        # JSON parse cəhdi
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            data = json.loads(json_str)
            return data.get("recommendations", [])
        return []
    except Exception as e:
        st.error(f"Süni intellekt xətası: {str(e)}")
        return []


def show_user_360(df: pd.DataFrame, company_map: pd.DataFrame, rec_df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Filterlərə görə uyğun müştəri tapılmadı.")
        return

    user = df.sort_values("total_customer_value", ascending=False).iloc[0]
    phone = user["phone_number"]
    
    # Müştəri dəyişəndə köhnə AI nəticələrini təmizləyirik (Reset state)
    if st.session_state.get("current_profile_phone") != phone:
        st.session_state["current_profile_phone"] = phone
        st.session_state["generative_recs"] = None

    active_companies = company_map[company_map["phone_number"] == phone]["company"].tolist()

    st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.6], gap="large")
    
    with left_col:
        st.markdown('<div class="panel" style="height: 100%;">', unsafe_allow_html=True)
        # Profile Header
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
            <div style="width:64px; height:64px; border-radius:12px; background:linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid rgba(148,163,184,0.2); display:flex; align-items:center; justify-content:center; color:#94a3b8; font-weight:700; font-size:24px;">
                ID
            </div>
            <div>
                <h2 style="margin:0; color:#f8fafc; font-size:1.6rem; letter-spacing:0.5px;">{phone}</h2>
                <div style="color:#94a3b8; font-size:14px; font-weight:600; margin-top:3px;">
                    {user['behavior_persona']} &nbsp;•&nbsp; Seqment: <span style="color:#38bdf8;">{user['segment']}</span>
                </div>
            </div>
        </div>
        <hr style="border-color: rgba(148,163,184,0.1); margin: 20px 0;"/>
        """, unsafe_allow_html=True)
        
        # Key Stats
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-bottom:25px; padding: 16px; background: rgba(15, 23, 42, 0.4); border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.1);">
            <div style="text-align: left; width: 50%; border-right: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="color:#64748b; font-size:12px; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">CLTV Dəyəri</div>
                <div style="font-size:22px; font-weight:800; color:#10b981;">${float(user['total_customer_value']):,.0f}</div>
            </div>
            <div style="text-align: right; width: 50%;">
                <div style="color:#64748b; font-size:12px; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Tərketmə Riski</div>
                <div style="font-size:22px; font-weight:800; color:#ef4444;">{user['churn_tier']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Compact Activity Bars
        def progress_html(label, val, color):
            return f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; font-size:14px; font-weight:600; margin-bottom:6px; color:#cbd5e1;">
                    <span>{label}</span>
                    <span style="color:{color};">{int(val)} / 100</span>
                </div>
                <div style="width:100%; background:rgba(255,255,255,0.05); border-radius:6px; height:10px; overflow:hidden;">
                    <div style="width:{min(100, max(0, val))}%; background:{color}; height:10px; border-radius:6px; box-shadow: 0 0 10px {color}66;"></div>
                </div>
            </div>
            """

        st.markdown(progress_html("Səyahət Aktivliyi (AzAL/ADY)", user["travel_activity"] * 1.7, "#10b981"), unsafe_allow_html=True)
        st.markdown(progress_html("Rəqəmsal (İnternet/Data)", user["digital_activity"] * 0.8, "#0ea5e9"), unsafe_allow_html=True)
        st.markdown(progress_html("Sadiqlik və Təkrar İstif.", 100 - user["churn_score"] * 100, "#f59e0b"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


    with right_col:
        st.markdown('<div class="panel" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#f8fafc; margin-top:0; margin-bottom:5px; font-size: 1.4rem;'>Təklif olunan Fəaliyyət Planı (Next Best Action)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:14px; margin-bottom: 20px;'>Müştəri davranışına əsaslanan fərdi saxlama və çarpaz satış tapşırıqları.</p>", unsafe_allow_html=True)
        
        generate_ai = st.button("Süni İntellektlə Yarat")
            
        if "generative_recs" not in st.session_state:
            st.session_state["generative_recs"] = None
            
        if generate_ai:
            with st.spinner("Analiz olunur..."):
                recs = generate_ai_recommendations(user)
                if recs:
                    st.session_state["generative_recs"] = recs
                else:
                    st.warning("Təkliflər formalaşdırılmadı.")

        if st.session_state["generative_recs"]:
            for r in st.session_state["generative_recs"]:
                st.markdown(f"""
                <div style="background:rgba(15, 23, 42, 0.4); border:1px solid rgba(52, 211, 153, 0.4); border-left:4px solid #10b981; padding:16px; border-radius:10px; margin-bottom:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <div style="color:#f8fafc; font-size:16px; font-weight:700;">{r.get('offer', 'Təklif')}</div>
                        <span style="background:rgba(14, 165, 233, 0.15); color:#38bdf8; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:700; border: 1px solid rgba(56, 189, 248, 0.3);">Hədəf: {r.get('target_company', 'Şirkət')}</span>
                    </div>
                    <div style="color:#cbd5e1; font-size:14px; line-height: 1.5; margin-bottom: 12px;">
                        <span style="color:#94a3b8; font-weight:600;">Səbəb:</span> {r.get('trigger', 'Səbəb yoxdur')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Fərdi kampaniya planı üçün 'Süni İntellektlə Yarat' düyməsinə klikləyin.")
        
        st.markdown("</div>", unsafe_allow_html=True)


def show_overview(df: pd.DataFrame, company_map: pd.DataFrame) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_kpis(df)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if not df.empty:
            fig = px.scatter(
                df,
                x="churn_score",
                y="total_customer_value",
                color="segment",
                size="cross_sell_potential",
                hover_name="phone_number",
                title="Müdaxilə Xəritəsi: Dəyər vs Tərketmə Riski",
                labels={"churn_score": "Tərketmə Riski (0-1)", "total_customer_value": "Müştəri Dəyəri ($)", "cross_sell_potential": "Çarpaz Şans"},
            )
            if df["total_customer_value"].max() > 0:
                fig.add_shape(
                    type="rect", 
                    x0=0.7, x1=1.0, 
                    y0=df["total_customer_value"].quantile(0.6) if len(df) > 1 else 0, 
                    y1=df["total_customer_value"].max() * 1.1, 
                    fillcolor="rgba(239, 68, 68, 0.15)", line_width=0
                )
                fig.add_annotation(
                    x=0.85, y=df["total_customer_value"].max(), 
                    text="Təcili Aksiyalar: Yüksək Dəyər + Risklilər", showarrow=False, font=dict(color="#ef4444")
                )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=58, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if df.empty:
            st.info("Filterlərə uyğun məlumat yoxdur.")
        else:
            opp_df = df.groupby("behavior_persona", as_index=False).agg(
                avg_cross_sell=("cross_sell_potential", "mean"),
                customer_count=("phone_number", "count")
            ).sort_values("avg_cross_sell", ascending=False)
            
            fig2 = px.bar(
                opp_df,
                y="behavior_persona",
                x="avg_cross_sell",
                color="customer_count",
                orientation="h",
                title="Kampaniya Hədəfləri (Personalar Üzrə Potensial)",
                labels={"behavior_persona": "Persona", "avg_cross_sell": "Ortalama Təklif Qəbulu Şansı", "customer_count": "Kütlə Böyüklüyü"},
                color_continuous_scale="Mint"
            )
            fig2.update_layout(height=400, margin=dict(l=10, r=10, t=58, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def show_unified_profiles(df: pd.DataFrame) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h3 style='margin-top:0; color:#f8fafc;'>Süzgəcdən Keçirilmiş Müştəri Siyahısı</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:14px;'>Axtarış filtrlərinə (Seqment, Risk, Persona) cavab verən tam müştəri bazası.</p>", unsafe_allow_html=True)
    with col2:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Detallı Baza (CSV) İndir",
            data=csv,
            file_name='secilmis_musteriler_bazasi.csv',
            mime='text/csv',
            use_container_width=True
        )

    cols = [
        "phone_number",
        "segment",
        "churn_tier",
        "total_customer_value",
        "cross_sell_potential",
        "behavior_persona",
        "company_count",
    ]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy().sort_values("total_customer_value", ascending=False)

    st.dataframe(view, use_container_width=True, hide_index=True, height=500)

    st.markdown("</div>", unsafe_allow_html=True)


def show_recommendations(df: pd.DataFrame, rec_df: pd.DataFrame) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Actionable Recommendation Layer")

    if rec_df.empty:
        st.warning("recommendations.csv tapılmadı və ya boşdur.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rec_filtered = rec_df[rec_df["phone_number"].isin(df["phone_number"])].copy()
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        top_actions = (
            rec_filtered.groupby(["target_company", "offer"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .head(12)
        )
        fig = px.bar(
            top_actions,
            x="count",
            y="offer",
            color="target_company",
            orientation="h",
            title="Top Recommended Actions",
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=58, b=10), yaxis_title="", xaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        merge_base = df[["phone_number", "total_customer_value"]].copy()
        rec_enriched = rec_filtered.merge(merge_base, on="phone_number", how="left")
        best = rec_enriched.sort_values("total_customer_value", ascending=False).head(12)
        st.dataframe(
            best[["phone_number", "target_company", "offer", "trigger", "total_customer_value"]],
            use_container_width=True,
            hide_index=True,
            height=430,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def show_network_graph(df: pd.DataFrame, company_map: pd.DataFrame) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Identity Link Graph")
    st.caption("Mavi düyünlər müştəridir, yaşıl düyünlər şirkət sistemləridir.")

    phones = df["phone_number"].head(120).tolist()
    edges_df = company_map[company_map["phone_number"].isin(phones)].copy()

    if edges_df.empty:
        st.info("Seçilən filterlərə görə qraf üçün data yoxdur.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    customers = sorted(edges_df["phone_number"].unique().tolist())
    companies = sorted(edges_df["company"].unique().tolist())

    if HAS_NETWORKX:
        g = nx.Graph()
        g.add_nodes_from(customers, node_type="customer")
        g.add_nodes_from(companies, node_type="company")
        g.add_edges_from(edges_df[["phone_number", "company"]].itertuples(index=False, name=None))
        pos = nx.spring_layout(g, seed=42, k=0.7)
    else:
        pos = {}
        for i, c in enumerate(companies):
            pos[c] = (0.0, float(i) * 2.0)
        for j, p in enumerate(customers):
            pos[p] = (5.0, float(j % max(6, len(companies) * 2)) * 0.8)

    edge_x, edge_y = [], []
    for _, row in edges_df.iterrows():
        x0, y0 = pos[row["phone_number"]]
        x1, y1 = pos[row["company"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.0, color="rgba(100,116,139,0.45)"),
        hoverinfo="none",
    )

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for n in customers + companies:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        if n in companies:
            node_color.append("#10b981")
            node_size.append(18)
            node_text.append(f"Company: {n}")
        else:
            node_color.append("#2563eb")
            node_size.append(9)
            node_text.append(f"Customer: {n}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#ffffff")),
        text=node_text,
        hovertemplate="%{text}<extra></extra>",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()

    data = load_all_data()
    unified = build_unified_profiles(data)
    company_map = available_companies_by_phone(data)
    coverage = company_map.groupby("phone_number", as_index=False).agg(company_count=("company", "nunique"))
    unified = unified.merge(coverage, on="phone_number", how="left")
    unified["company_count"] = unified["company_count"].fillna(1).astype(int)

    render_top_nav()
    filters = sidebar_filters(unified)
    filtered = apply_filters(unified, filters)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Analitika Xülasəsi", 
        "Fərdi Crm Profili", 
        "Ümumi Kampaniyalar",
        "Müştəri Siyahısı (Export)"
    ])

    with tab1:
        show_overview(filtered, company_map)
        
    with tab2:
        st.markdown("<h3 style='color:#38bdf8;'>Müştəri İzləyici Paneli</h3>", unsafe_allow_html=True)
        top_users = filtered.sort_values("total_customer_value", ascending=False)["phone_number"].head(50).tolist()
        if top_users:
            selected_user = st.selectbox("Müştərini seçin", top_users)
            user_df = filtered[filtered["phone_number"] == selected_user]
            show_user_360(user_df, company_map, data.get("recommendations", pd.DataFrame()))
        else:
            st.info("Filtrə uyğun müştəri tapılmadı.")

    with tab3:
        show_recommendations(filtered, data.get("recommendations", pd.DataFrame()))

    with tab4:
        show_unified_profiles(filtered)


if __name__ == "__main__":
    main()