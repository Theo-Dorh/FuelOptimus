"""
FuelOptimus | Pricing Decision Support
--------------------------------------
Streamlit front end for the FuelOptimus two-stage pricing engine.

Stage 1  Volume model, trained at startup from ICON_MAT_Clean.csv using the
         exact pipeline in FuelOptimus_Colab_v2.py (Cells 2, 4, 5, 6, 7):
         lag/rolling features per Station_ID x Product, chronological split
         (train <= 2025-09-30, val <= 2026-03-15, test after), MinMaxScaler
         fitted on train only, GridSearchCV + TimeSeriesSplit(5) for Ridge,
         Random Forest and XGBoost, best model chosen by test R².
Stage 2  Grid search over premiums 0.00-2.00 GHS/L (step 0.01), margin =
         delta x predicted volume, constrained to delta <= competitor cap.

Run:  streamlit run app.py
(ICON_MAT_Clean.csv must sit in the same folder as app.py.)
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "ICON_MAT_Clean.csv")

SEED = 42
TRAIN_END = pd.Timestamp("2025-09-30")
VAL_END = pd.Timestamp("2026-03-15")

DELTA_MIN, DELTA_MAX, DELTA_STEP = 0.00, 2.00, 0.01

FEATURES = [
    "Price_Premium", "PP_Lag1", "PP_Lag7", "PP_Lag14",
    "Vol_Roll_Mean_7", "Vol_Roll_Mean_14", "Vol_Roll_Mean_30", "Vol_Roll_Std_7",
    "Margin_Roll_Mean_14",
    "Comp_Gap", "Above_Market", "Comp_Roll_7", "PP_Percentile_30",
    "NPA_Price_Floor", "Window_Number", "Days_Since_Window_Start",
    "Regulatory_Period", "Is_Weekend", "Month_Number",
    "Region_Enc", "Product_Enc", "Station_Enc", "DOW_Enc",
    "Price_Imputed", "Volume_Imputed", "Comp_Imputed",
]
TARGET = "Volume_Sold"

PRODUCTS = {"PMS (Premium Gasoline)": "PMS", "AGO (Diesel)": "AGO"}

# Palette (wireframe + brief)
NAVY_DARK = "#0D1B3E"
NAVY = "#1B3A6B"
RED = "#C0392B"
GREEN = "#1A7A4A"
GREY = "#6B7280"
LINE = "#D5DCE8"
TILE = "#F2F5FA"
INK = "#1F2937"
MUTED = "#6B7385"


# ════════════════════════════════════════════════════════════
# STAGE 1 — TRAINING (replicates the Colab notebook)
# ════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame):
    """Colab Cell 4, unchanged in logic."""
    df = df.copy().sort_values(["Station_ID", "Product", "Date"])
    grp = df.groupby(["Station_ID", "Product"])

    df["PP_Lag1"] = grp["Price_Premium"].shift(1)
    df["PP_Lag7"] = grp["Price_Premium"].shift(7)
    df["PP_Lag14"] = grp["Price_Premium"].shift(14)

    df["Vol_Roll_Mean_7"] = grp["Volume_Sold"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    df["Vol_Roll_Mean_14"] = grp["Volume_Sold"].transform(
        lambda x: x.shift(1).rolling(14, min_periods=5).mean())
    df["Vol_Roll_Mean_30"] = grp["Volume_Sold"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    df["Vol_Roll_Std_7"] = grp["Volume_Sold"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).std()).fillna(0)

    df["Margin_Roll_Mean_14"] = grp["Margin_Contribution"].transform(
        lambda x: x.shift(1).rolling(14, min_periods=5).mean())

    df["Above_Market"] = (df["Pump_Price"] > df["Comp_Mean"]).astype(int)
    df["Comp_Roll_7"] = grp["Comp_Mean"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    df["PP_Percentile_30"] = grp["Price_Premium"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).rank(pct=True)).fillna(0.5)

    le_region, le_product = LabelEncoder(), LabelEncoder()
    le_station, le_dow = LabelEncoder(), LabelEncoder()
    df["Region_Enc"] = le_region.fit_transform(df["Region"])
    df["Product_Enc"] = le_product.fit_transform(df["Product"])
    df["Station_Enc"] = le_station.fit_transform(df["Station_ID"])
    df["DOW_Enc"] = le_dow.fit_transform(df["Day_of_Week"])
    return df, le_region, le_product, le_station


@dataclass
class Engine:
    model: object
    best_name: str
    scaler: MinMaxScaler
    features: list
    df_model: pd.DataFrame          # engineered rows with complete features
    performance: pd.DataFrame       # test-set metrics for all three models
    data_end: pd.Timestamp


def _load_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Station_ID", "Region", "Product", "Day_of_Week", "Pricing_Window"]:
        df[col] = df[col].astype(str)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_engine(csv_bytes: bytes) -> Engine:
    """Colab Cells 2, 4, 5, 6, 7. Cached: runs once per server process."""
    df = _load_csv(io.BytesIO(csv_bytes))
    df_feat, *_ = engineer_features(df)
    df_model = df_feat.dropna(subset=FEATURES).copy()

    train = df_model[df_model["Date"] <= TRAIN_END]
    test = df_model[df_model["Date"] > VAL_END]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=5)
    scoring = "neg_root_mean_squared_error"

    ridge = GridSearchCV(Ridge(random_state=SEED), {"alpha": [0.1, 1.0, 10.0]},
                         cv=tscv, scoring=scoring, n_jobs=-1).fit(X_train_s, y_train)
    rf = GridSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [10, None],
         "min_samples_split": [2, 5], "max_features": ["sqrt"]},
        cv=tscv, scoring=scoring, n_jobs=-1).fit(X_train_s, y_train)
    xg = GridSearchCV(
        xgb.XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist",
                         eval_metric="rmse"),
        {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
         "max_depth": [5, 7], "subsample": [0.8], "colsample_bytree": [0.8],
         "reg_lambda": [1, 5]},
        cv=tscv, scoring=scoring, n_jobs=-1).fit(X_train_s, y_train)

    candidates = [("Ridge", ridge.best_estimator_),
                  ("Random Forest", rf.best_estimator_),
                  ("XGBoost", xg.best_estimator_)]
    rows = []
    for name, m in candidates:
        p = m.predict(X_test_s)
        rows.append({"Model": name,
                     "MAE (L)": mean_absolute_error(y_test, p),
                     "RMSE (L)": np.sqrt(mean_squared_error(y_test, p)),
                     "R²": r2_score(y_test, p)})
    perf = pd.DataFrame(rows)
    best_idx = int(perf["R²"].idxmax())
    best_name, best_model = candidates[best_idx]

    return Engine(model=best_model, best_name=best_name, scaler=scaler,
                  features=FEATURES, df_model=df_model, performance=perf,
                  data_end=df["Date"].max())


# ════════════════════════════════════════════════════════════
# STAGE 2 — MARGIN OPTIMISATION
# ════════════════════════════════════════════════════════════

def window_label(code: str) -> str:
    """'2026JunW2' -> '2026 Jun W2'"""
    return f"{code[:4]} {code[4:7]} {code[7:]}" if len(code) >= 9 else code


def base_row(engine: Engine, station: str, product: str, window: str) -> pd.Series:
    """Most recent complete feature row for this station/product in the chosen
    window; falls back to the latest row before the window, then the latest row."""
    sp = engine.df_model[(engine.df_model["Station_ID"] == station)
                         & (engine.df_model["Product"] == product)]
    if sp.empty:
        return None
    in_win = sp[sp["Pricing_Window"] == window]
    if not in_win.empty:
        return in_win.sort_values("Date").iloc[-1]
    win_start = engine.df_model.loc[engine.df_model["Pricing_Window"] == window, "Date"].min()
    before = sp[sp["Date"] < win_start] if pd.notna(win_start) else sp.iloc[0:0]
    return (before if not before.empty else sp).sort_values("Date").iloc[-1]


def _rows_for_deltas(row: pd.Series, deltas: np.ndarray, npa_floor: float,
                     comp_mean: float) -> pd.DataFrame:
    """Colab find_optimal_premium() row mutation, vectorised."""
    X = pd.DataFrame([row[FEATURES].astype(float)] * len(deltas)).reset_index(drop=True)
    X["NPA_Price_Floor"] = npa_floor
    X["Price_Premium"] = deltas
    X["Comp_Gap"] = (npa_floor + deltas) - comp_mean
    X["Above_Market"] = (X["Comp_Gap"] > 0).astype(int)
    return X[FEATURES]


def predict_volume(engine: Engine, X: pd.DataFrame) -> np.ndarray:
    return np.maximum(engine.model.predict(engine.scaler.transform(X)), 0)


def optimise(engine: Engine, row: pd.Series, npa_floor: float, comp_mean: float,
             current_premium: float) -> dict:
    deltas = np.round(np.arange(DELTA_MIN, DELTA_MAX + DELTA_STEP / 2, DELTA_STEP), 2)
    vols = predict_volume(engine, _rows_for_deltas(row, deltas, npa_floor, comp_mean))
    margins = deltas * vols

    cap = max(comp_mean - npa_floor, 0.0)
    feasible = deltas <= cap + 1e-9
    i_opt = int(np.argmax(np.where(feasible, margins, -np.inf)))
    d_opt, v_opt, m_opt = float(deltas[i_opt]), float(vols[i_opt]), float(margins[i_opt])

    v_cur = float(predict_volume(engine, _rows_for_deltas(
        row, np.array([current_premium]), npa_floor, comp_mean))[0])
    m_cur = current_premium * v_cur

    # Unconstrained optimum, for the "capped" flag
    i_free = int(np.argmax(margins))
    capped = bool(cap > 0 and deltas[i_free] > d_opt + 1e-9)   # constraint is binding

    return {
        "curve": pd.DataFrame({"Premium_GHS_per_L": deltas,
                               "Pred_Volume_L_per_day": vols,
                               "Exp_Margin_GHS_per_day": margins,
                               "Feasible": feasible}),
        "cap": cap, "delta_opt": d_opt, "vol_opt": v_opt, "margin_opt": m_opt,
        "current_premium": current_premium, "vol_cur": v_cur, "margin_cur": m_cur,
        "pump_price": npa_floor + d_opt, "capped": bool(capped),
        "no_headroom": cap <= 0,
    }


def explain(engine: Engine, row: pd.Series, station: str, product: str,
            npa_floor: float, comp_mean: float, delta: float, top_n: int = 6) -> pd.DataFrame:
    """Local drivers of the prediction at delta*.

    Reference point is this station-product's average historical day. Each
    feature's effect = prediction(actual inputs) - prediction(with that one
    feature reset to its reference value). For the Ridge engine this equals
    coef_j * (x_j - ref_j) in scaled space exactly. Effects are divided by the
    station-product's mean volume, giving a unitless volume-index contribution.
    """
    sp = engine.df_model[(engine.df_model["Station_ID"] == station)
                         & (engine.df_model["Product"] == product)]
    ref = sp[FEATURES].astype(float).mean()
    x = _rows_for_deltas(row, np.array([delta]), npa_floor, comp_mean).iloc[0]

    probes = [x.copy()]
    for f in FEATURES:
        p = x.copy()
        p[f] = ref[f]
        probes.append(p)
    preds = engine.model.predict(engine.scaler.transform(pd.DataFrame(probes)[FEATURES]))
    effects = (preds[0] - preds[1:]) / max(sp[TARGET].mean(), 1.0)

    out = pd.DataFrame({"Feature": FEATURES, "Effect": effects})
    out = out.reindex(out["Effect"].abs().sort_values(ascending=False).index).head(top_n)
    return out.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════

FONT = "Calibri, 'Segoe UI', Lato, system-ui, -apple-system, sans-serif"


def margin_chart(res: dict) -> go.Figure:
    c = res["curve"]
    cap = res["cap"]
    inside = c[c["Feasible"]]
    outside = c[c["Premium_GHS_per_L"] >= inside["Premium_GHS_per_L"].max() - 1e-9]
    ymax = max(c["Exp_Margin_GHS_per_day"].max(), res["margin_cur"], 1) * 1.12

    fig = go.Figure()
    if cap < DELTA_MAX:
        fig.add_vrect(x0=cap, x1=DELTA_MAX, fillcolor=RED, opacity=0.2,
                      line_width=0, layer="below")
        fig.add_vline(x=cap, line=dict(color=RED, width=2.5, dash="dash"))
        fig.add_annotation(x=(cap + DELTA_MAX) / 2, y=ymax * 0.06,
                           text="above local market:<br>excluded", showarrow=False,
                           font=dict(color=RED, size=12, family=FONT))
        fig.add_trace(go.Scatter(
            x=outside["Premium_GHS_per_L"], y=outside["Exp_Margin_GHS_per_day"],
            mode="lines", line=dict(color="#8A8FA3", width=2.5, dash="dash"),
            hovertemplate="δ %{x:.2f}<br>GHS %{y:,.0f}/day<extra>excluded</extra>"))
    fig.add_trace(go.Scatter(
        x=inside["Premium_GHS_per_L"], y=inside["Exp_Margin_GHS_per_day"],
        mode="lines", line=dict(color=NAVY, width=4),
        hovertemplate="δ %{x:.2f}<br>GHS %{y:,.0f}/day<extra></extra>"))

    # current premium
    if 0 <= res["current_premium"] <= DELTA_MAX:
        fig.add_trace(go.Scatter(
            x=[res["current_premium"]], y=[res["margin_cur"]], mode="markers+text",
            marker=dict(color=GREY, size=13), text=["current"],
            textposition="bottom right", textfont=dict(color=GREY, size=13, family=FONT),
            hovertemplate="current δ %{x:.2f}<br>GHS %{y:,.0f}/day<extra></extra>"))
    # optimum
    fig.add_trace(go.Scatter(
        x=[res["delta_opt"]], y=[res["margin_opt"]], mode="markers+text",
        marker=dict(color=RED, size=15, line=dict(color="white", width=2)),
        text=["<b>δ*</b>"], textposition="top left",
        textfont=dict(color=RED, size=16, family=FONT),
        hovertemplate="δ* %{x:.2f}<br>GHS %{y:,.0f}/day<extra></extra>"))

    axis = dict(showgrid=False, zeroline=False, showline=True, linecolor="#111",
                ticks="outside", tickcolor="#111", tickfont=dict(size=12, color="#111"))
    fig.update_layout(
        template="plotly_white", height=360, margin=dict(l=10, r=10, t=6, b=10), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family=FONT),
        xaxis=dict(**axis, range=[0, DELTA_MAX], dtick=0.5, tickformat=".1f",
                   title=dict(text="δ (GHS/L)", font=dict(size=13, color="#111"))),
        yaxis=dict(**axis, range=[0, ymax], tickformat=",.0f", nticks=4,
                   title=dict(text="GHS/day", font=dict(size=13, color="#111"))),
        hoverlabel=dict(font_family=FONT),
    )
    return fig


def drivers_chart(drv: pd.DataFrame) -> go.Figure:
    d = drv.iloc[::-1]  # largest at top
    lim = max(d["Effect"].abs().max(), 1e-6) * 1.45
    fig = go.Figure(go.Bar(
        x=d["Effect"], y=d["Feature"], orientation="h",
        marker_color=[NAVY if v >= 0 else RED for v in d["Effect"]],
        text=[f"{v:+.3f}" for v in d["Effect"]], textposition="outside",
        textfont=dict(size=12, color="#111", family=FONT), cliponaxis=False,
        width=0.6, hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#444", width=1.5))
    fig.update_layout(
        template="plotly_white", height=334, margin=dict(l=10, r=40, t=6, b=10), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family=FONT),
        xaxis=dict(visible=False, range=[-lim, lim]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#111"),
                   ticksuffix="   "),
    )
    return fig


# ════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════

def export_csv(ctx: dict, res: dict, drv: pd.DataFrame) -> bytes:
    summary = [
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Engine", ctx["engine_name"]),
        ("Station", ctx["station"]), ("Region", ctx["region"]),
        ("Product", ctx["product"]), ("NPA window", ctx["window"]),
        ("NPA floor (GHS/L)", f"{ctx['floor']:.2f}"),
        ("Competitor mean (GHS/L)", f"{ctx['comp']:.2f}"),
        ("Current premium (GHS/L)", f"{res['current_premium']:.2f}"),
        ("Competition cap (GHS/L)", f"{res['cap']:.2f}"),
        ("Recommended premium (GHS/L)", f"{res['delta_opt']:.2f}"),
        ("Pump price (GHS/L)", f"{res['pump_price']:.2f}"),
        ("Expected volume (L/day)", f"{res['vol_opt']:.0f}"),
        ("Expected margin (GHS/day)", f"{res['margin_opt']:.0f}"),
        ("Margin at current premium (GHS/day)", f"{res['margin_cur']:.0f}"),
        ("Improvement vs current (GHS/day)", f"{res['margin_opt'] - res['margin_cur']:.0f}"),
        ("Capped at competitor mean", "Yes" if res["capped"] else "No"),
    ]
    buf = io.StringIO()
    buf.write("FuelOptimus recommendation\n")
    pd.DataFrame(summary, columns=["Item", "Value"]).to_csv(buf, index=False)
    buf.write("\nDrivers of predicted volume index\n")
    drv.round(4).to_csv(buf, index=False)
    buf.write("\nCandidate premium grid\n")
    res["curve"].round(3).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_pdf(ctx: dict, res: dict, drv: pd.DataFrame) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = res["curve"]
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.94, "FuelOptimus", fontsize=20, weight="bold", color=NAVY)
    fig.text(0.20, 0.94, "Pricing recommendation", fontsize=14, color=MUTED)
    fig.text(0.96, 0.94, datetime.now().strftime("%d %b %Y %H:%M"),
             fontsize=10, color=MUTED, ha="right")
    fig.text(0.04, 0.895,
             f"{ctx['station']} · {ctx['region']}   |   {ctx['product']}   |   "
             f"NPA window {ctx['window']}   |   {ctx['engine_name']} engine",
             fontsize=10.5, color=INK)

    lines = [
        ("Recommended premium", f"GHS {res['delta_opt']:.2f} / litre", NAVY),
        ("Pump price", f"GHS {res['pump_price']:.2f}", INK),
        ("Expected volume", f"{res['vol_opt']:,.0f} L/day", NAVY),
        ("Expected margin", f"GHS {res['margin_opt']:,.0f}/day", NAVY),
        ("vs current pricing", f"{res['margin_opt'] - res['margin_cur']:+,.0f} GHS/day",
         GREEN if res['margin_opt'] >= res['margin_cur'] else RED),
        ("NPA floor / competitor mean",
         f"GHS {ctx['floor']:.2f} / {ctx['comp']:.2f}", INK),
        ("Current premium", f"GHS {res['current_premium']:.2f}", INK),
        ("Competition cap", f"GHS {res['cap']:.2f}", INK),
    ]
    y = 0.83
    for k, v, col in lines:
        fig.text(0.04, y, k, fontsize=10.5, color=MUTED)
        fig.text(0.26, y, v, fontsize=11, weight="bold", color=col)
        y -= 0.04
    if res["capped"]:
        fig.text(0.04, y, "capped at local competitor mean", fontsize=10.5, color=RED)

    ax = fig.add_axes([0.42, 0.50, 0.53, 0.36])
    ins = c[c["Feasible"]]
    ax.plot(ins["Premium_GHS_per_L"], ins["Exp_Margin_GHS_per_day"], color=NAVY, lw=2.5)
    if res["cap"] < DELTA_MAX:
        out = c[c["Premium_GHS_per_L"] >= ins["Premium_GHS_per_L"].max()]
        ax.plot(out["Premium_GHS_per_L"], out["Exp_Margin_GHS_per_day"],
                color="#8A8FA3", lw=2, ls="--")
        ax.axvspan(res["cap"], DELTA_MAX, color=RED, alpha=0.18, lw=0)
        ax.axvline(res["cap"], color=RED, ls="--", lw=1.8)
    if 0 <= res["current_premium"] <= DELTA_MAX:
        ax.scatter([res["current_premium"]], [res["margin_cur"]], color=GREY, s=60, zorder=5)
        ax.annotate("current", (res["current_premium"], res["margin_cur"]),
                    xytext=(8, -14), textcoords="offset points", color=GREY)
    ax.scatter([res["delta_opt"]], [res["margin_opt"]], color=RED, s=80, zorder=6)
    ax.annotate("δ*", (res["delta_opt"], res["margin_opt"]), xytext=(-18, 8),
                textcoords="offset points", color=RED, weight="bold", fontsize=13)
    ax.set_xlim(0, DELTA_MAX); ax.set_ylim(bottom=0)
    ax.set_xlabel("δ (GHS/L)"); ax.set_ylabel("GHS/day")
    ax.set_title("Margin vs premium", loc="left", color=NAVY, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = fig.add_axes([0.56, 0.08, 0.39, 0.32])
    d = drv.iloc[::-1]
    ax2.barh(d["Feature"], d["Effect"],
             color=[NAVY if v >= 0 else RED for v in d["Effect"]], height=0.6)
    ax2.axvline(0, color="#444", lw=1)
    lim = max(d["Effect"].abs().max(), 1e-6) * 1.45
    ax2.set_xlim(-lim, lim)
    for yi, v in enumerate(d["Effect"]):
        ax2.text(v + (lim * 0.03 if v >= 0 else -lim * 0.03), yi, f"{v:+.3f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax2.set_xticks([])
    ax2.set_title("Why this price — drivers of predicted volume index",
                  loc="left", color=NAVY, weight="bold", fontsize=11)
    for s in ("top", "right", "bottom"):
        ax2.spines[s].set_visible(False)

    fig.text(0.04, 0.03, "Decision support only. Predictions from a volume model trained "
             "on historical station data; final pricing remains a management decision.",
             fontsize=8, color=MUTED)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

st.set_page_config(page_title="FuelOptimus | Pricing Decision Support",
                   page_icon="⛽", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
html, body, [class*="css"], .stApp, button, input, select, textarea {{
    font-family: {FONT} !important;
}}
.stApp {{ background: #F8FAFC; }}
header[data-testid="stHeader"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1500px; }}

/* Header bar */
.fo-header {{
    background: {NAVY}; color: #fff; border-radius: 14px;
    padding: 20px 30px; display: flex; align-items: baseline; gap: 34px;
    margin-bottom: 18px;
}}
.fo-brand {{ font-size: 30px; font-weight: 700; letter-spacing: .2px; }}
.fo-sub {{ font-size: 22px; color: #DCE4F2; }}
.fo-ver {{ margin-left: auto; font-size: 17px; color: #DCE4F2; }}

/* Bordered panels */
.st-key-inputs_panel, .st-key-chart_panel, .st-key-why_panel {{
    background: #fff; border: 2px solid {LINE}; border-radius: 14px;
    padding: 18px 20px 20px 20px;
}}
.st-key-rec_panel {{
    background: #fff; border: 3px solid {NAVY}; border-radius: 14px;
    padding: 18px 26px 18px 26px; min-height: 248px;
}}

.fo-title {{ color: {NAVY}; font-size: 21px; font-weight: 700; margin: 0 0 4px 0; }}
.fo-subtitle {{ color: {MUTED}; font-size: 15px; margin: -2px 0 6px 0; }}
.fo-eyebrow {{ color: {MUTED}; font-size: 16px; font-weight: 700; margin: 2px 0 0 0; }}

/* Inputs */
[data-testid="stWidgetLabel"] p {{ color: {MUTED} !important; font-size: 15px !important; }}
.st-key-inputs_panel [data-baseweb="select"] > div,
.st-key-inputs_panel [data-baseweb="input"] {{
    background: {TILE} !important; border: 2px solid {LINE} !important; border-radius: 8px !important;
}}
.st-key-inputs_panel [data-baseweb="input"] input {{ background: {TILE} !important; font-size: 16px; }}
.st-key-inputs_panel [data-testid="stNumberInputStepDown"],
.st-key-inputs_panel [data-testid="stNumberInputStepUp"] {{ display: none; }}

.st-key-inputs_panel [data-baseweb="select"] div {{ font-size: 16px !important; }}

/* Buttons */
.st-key-gen_btn button {{
    background: {NAVY} !important; color: #fff !important; border: none !important;
    border-radius: 9px !important; height: 52px; font-size: 17px !important; font-weight: 700 !important;
}}
.st-key-gen_btn button:hover {{ background: {NAVY_DARK} !important; }}
.st-key-gen_btn button p, .st-key-export_pop button p {{ font-size: 17px !important; font-weight: 700 !important; }}
.st-key-export_pop button {{
    background: #fff !important; color: {NAVY} !important; border: 2.5px solid {NAVY} !important;
    border-radius: 9px !important; height: 52px;
}}

/* Recommendation */
.fo-big {{ display: flex; align-items: baseline; gap: 22px; margin: 6px 0 10px 0; }}
.fo-big .v {{ color: {NAVY}; font-size: 52px; font-weight: 800; line-height: 1.05; }}
.fo-big .u {{ color: {NAVY}; font-size: 22px; }}
.fo-pump {{ color: {INK}; font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
.fo-cap {{ color: {RED}; font-size: 17px; min-height: 24px; }}

/* Metric tiles */
.fo-tile {{
    background: {TILE}; border-radius: 12px; padding: 20px 24px; margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
}}
.fo-tile .k {{ color: {MUTED}; font-size: 17px; }}
.fo-tile .v {{ font-size: 23px; font-weight: 800; }}

.fo-note {{ color: {MUTED}; font-size: 13px; }}
.fo-stale {{ color: {RED}; font-size: 13px; margin-top: 6px; }}
</style>
""", unsafe_allow_html=True)


# ── Data + training ────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error("ICON_MAT_Clean.csv was not found next to app.py.")
    up = st.file_uploader("Upload ICON_MAT_Clean.csv to continue", type="csv")
    if up is None:
        st.stop()
    csv_bytes = up.getvalue()
else:
    with open(DATA_PATH, "rb") as fh:
        csv_bytes = fh.read()

with st.spinner("Training the FuelOptimus engine from ICON_MAT_Clean.csv "
                "(Ridge, Random Forest, XGBoost with grid search). "
                "This happens once per session start and takes 1–3 minutes."):
    engine = train_engine(csv_bytes)

st.markdown(f"""
<div class="fo-header">
  <span class="fo-brand">FuelOptimus</span>
  <span class="fo-sub">Pricing Decision Support</span>
  <span class="fo-ver">v1.0 · {engine.best_name} engine · retrained {engine.data_end:%b %Y}</span>
</div>
""", unsafe_allow_html=True)

dm = engine.df_model
stations = (dm.groupby("Station_ID")["Region"].first().reset_index()
            .sort_values("Station_ID"))
station_labels = {f"{r.Station_ID} · {r.Region}": r.Station_ID
                  for r in stations.itertuples()}
win_order = (dm.groupby("Pricing_Window")["Date"].min().sort_values().index.tolist())

col_in, col_main = st.columns([1.05, 3.6], gap="medium")

# ── Left: pricing inputs ───────────────────────────────────
with col_in:
    with st.container(key="inputs_panel"):
        st.markdown('<p class="fo-title" style="margin-bottom:10px">PRICING INPUTS</p>',
                    unsafe_allow_html=True)
        st_label = st.selectbox("Station", list(station_labels.keys()))
        station = station_labels[st_label]
        region = st_label.split(" · ", 1)[1]
        prod_label = st.selectbox("Product", list(PRODUCTS.keys()))
        product = PRODUCTS[prod_label]
        window = st.selectbox("NPA window", win_order[::-1], format_func=window_label)

        row = base_row(engine, station, product, window)
        if row is None:
            st.warning(f"No {product} history for {station}.")
            st.stop()

        k = f"{station}_{product}_{window}"   # new defaults whenever selection changes
        floor = st.number_input("NPA floor price", min_value=0.0, step=0.01, format="%.2f",
                                value=float(row["NPA_Price_Floor"]), key=f"floor_{k}",
                                help="GHS per litre. Defaults to the NPA floor for this product and window.")
        comp = st.number_input("Competitor mean", min_value=0.0, step=0.01, format="%.2f",
                               value=float(row["Comp_Mean"]), key=f"comp_{k}",
                               help="GHS per litre. Auto-filled from Comp_Mean for this station and product.")
        cur = st.number_input("Current premium", min_value=0.0, step=0.01, format="%.2f",
                              value=float(row["Price_Premium"]), key=f"cur_{k}",
                              help="GHS per litre. The station's most recent actual premium.")
        st.markdown(f'<div class="fo-note">Values in GHS / L · base day '
                    f'{row["Date"]:%d %b %Y}</div>', unsafe_allow_html=True)
        st.write("")

        gen = st.button("Generate recommendation", key="gen_btn", width="stretch")
        export_slot = st.empty()

inputs = (station, product, window, round(floor, 4), round(comp, 4), round(cur, 4))

if gen or "result" not in st.session_state:
    res = optimise(engine, row, floor, comp, cur)
    drv = explain(engine, row, station, product, floor, comp, res["delta_opt"])
    st.session_state["result"] = {
        "inputs": inputs, "res": res, "drv": drv,
        "ctx": {"station": station, "region": region, "product": prod_label,
                "window": window_label(window), "floor": floor, "comp": comp,
                "engine_name": engine.best_name},
    }

R = st.session_state["result"]
res, drv, ctx = R["res"], R["drv"], R["ctx"]
stale = R["inputs"] != inputs

with export_slot.container():
    with st.popover("Export CSV / PDF", width="stretch", key="export_pop"):
        fname = f"FuelOptimus_{ctx['station'].split(' ')[0]}_{ctx['product'][:3]}_" \
                f"{ctx['window'].replace(' ', '')}"
        st.download_button("Download CSV", export_csv(ctx, res, drv),
                           file_name=f"{fname}.csv", mime="text/csv", width="stretch")
        st.download_button("Download PDF", export_pdf(ctx, res, drv),
                           file_name=f"{fname}.pdf", mime="application/pdf", width="stretch")
    if stale:
        st.markdown('<div class="fo-stale">Inputs changed. Click Generate to refresh '
                    'the recommendation.</div>', unsafe_allow_html=True)

# ── Centre + right ─────────────────────────────────────────
with col_main:
    top_l, top_r = st.columns([1.0, 1.1], gap="medium")

    with top_l:
        with st.container(key="rec_panel"):
            if res["no_headroom"]:
                cap_note = "competitor mean at or below NPA floor: no headroom"
            elif res["capped"]:
                cap_note = "capped at local competitor mean"
            else:
                cap_note = "&nbsp;"
            st.markdown(f"""
            <p class="fo-eyebrow">RECOMMENDED PREMIUM</p>
            <div class="fo-big"><span class="v">GHS {res['delta_opt']:.2f}</span>
                 <span class="u">/ litre</span></div>
            <div class="fo-pump">Pump price GHS {res['pump_price']:.2f}</div>
            <div class="fo-cap">{cap_note}</div>
            """, unsafe_allow_html=True)

    with top_r:
        gain = res["margin_opt"] - res["margin_cur"]
        gcol = GREEN if gain >= 0 else RED
        sign = "+" if gain >= 0 else "−"
        st.markdown(f"""
        <div class="fo-tile"><span class="k">Expected volume</span>
            <span class="v" style="color:{NAVY}">{res['vol_opt']:,.0f} L/day</span></div>
        <div class="fo-tile"><span class="k">Expected margin</span>
            <span class="v" style="color:{NAVY}">GHS {res['margin_opt']:,.0f}/day</span></div>
        <div class="fo-tile"><span class="k">vs current pricing</span>
            <span class="v" style="color:{gcol}">{sign}GHS {abs(gain):,.0f}/day</span></div>
        """, unsafe_allow_html=True)

    bot_l, bot_r = st.columns([1.1, 1.0], gap="medium")
    cfg = {"displayModeBar": False}
    with bot_l:
        with st.container(key="chart_panel"):
            st.markdown('<p class="fo-title">Margin vs premium</p>', unsafe_allow_html=True)
            st.plotly_chart(margin_chart(res), config=cfg, theme=None, key="margin_fig")
    with bot_r:
        with st.container(key="why_panel"):
            st.markdown('<p class="fo-title">Why this price</p>'
                        '<p class="fo-subtitle">drivers of predicted volume index</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(drivers_chart(drv), config=cfg, theme=None, key="why_fig")

with st.expander("Engine details"):
    perf = engine.performance.copy()
    perf["Selected"] = np.where(perf["Model"] == engine.best_name, "✓", "")
    st.dataframe(perf.style.format({"MAE (L)": "{:.1f}", "RMSE (L)": "{:.1f}", "R²": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.caption(
        f"Trained on {int((dm['Date'] <= TRAIN_END).sum()):,} rows up to {TRAIN_END:%d %b %Y}; "
        f"best model chosen by test-set R² (after {VAL_END:%d %b %Y}), as in the thesis notebook. "
        "Drivers compare this prediction with the station's average historical day, "
        "scaled by its mean daily volume.")
