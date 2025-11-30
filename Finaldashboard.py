# === Bintix Waste Analytics — CSV-only (works with your new finaldataset.csv) ===
# Same features & UI; no Parquet; Lat/Lon read from the CSV; works with 3 months/feature.

import re
import io
import base64
import mimetypes
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster,HeatMap
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib import colormaps
from branca.element import MacroElement
from jinja2 import Template


# ---------------- App & Brand ----------------
st.set_page_config(page_title="Bintix Waste Analytics", layout="wide")

###"#36204D"     # purple
BRAND_PRIMARY = 'purple'
TEXT_DARK = "#36204D"

# Speed settings
ST_MAP_HEIGHT = 900
ST_RETURNED_OBJECTS = []  # don't send all map layers back to Streamlit

# --- Environmental conversions ---
CO2_PER_KG_DRY = 2.18      # 1 kg dry waste -> 2.18 kg CO2 averted
KG_PER_TREE     = 117.0    # 117 kg dry waste -> 1 tree saved

# ---------------- Assets (icons) ----------------
BASE_DIR = Path(__file__).parent.resolve()
_ASSET_DIR_CANDIDATES = [BASE_DIR / "assets", BASE_DIR / "assests"]
ASSETS_DIR = next((p for p in _ASSET_DIR_CANDIDATES if p.exists()), _ASSET_DIR_CANDIDATES[0])

@st.cache_resource(show_spinner=False)
def load_icon_data_uri(filename: str) -> str:
    """Return a data: URI for an image in ASSETS_DIR so it renders inside Folium popups."""
    p = ASSETS_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Icon not found: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

try:
    TREE_ICON     = load_icon_data_uri("tree.png")
    HOUSE_ICON    = load_icon_data_uri("house.png")
    RECYCLE_ICON  = load_icon_data_uri("waste-management.png")
except FileNotFoundError as e:
    st.error(f"{e}\nMake sure your icons are in: {ASSETS_DIR}")
    TREE_ICON = HOUSE_ICON = RECYCLE_ICON = ""

# ---------------- Data loading (CSV only) ----------------
# Your new dataset path
CSV_WINDOWS_DEFAULT = Path(r"C:\Users\meena\Downloads\kaggle\FINALWORK\book2.csv")
# Fallback to a local copy if you drop it next to the app
CSV_LOCAL_FALLBACK  = BASE_DIR / "book2.csv"
CSV_DEFAULT = CSV_WINDOWS_DEFAULT if CSV_WINDOWS_DEFAULT.exists() else CSV_LOCAL_FALLBACK

ID_COLS_REQUIRED = ["City", "Community", "Pincode"]
ID_COLS_OPTIONAL = ["Lat", "Lon"]  # we will normalize lat/lon naming to these
METRIC_COL_REGEX = re.compile(
    r"^(Tonnage|CO2_Kgs_Averted|Households_Participating|Segregation_Compliance_Pct|Trees_Saved)_(\d{4}-\d{2})$"
)

def _detect_metric_month_cols(columns):
    cols, months = [], set()
    for c in columns:
        m = METRIC_COL_REGEX.match(c)
        if m:
            cols.append(c); months.add(m.group(2))
    return cols, sorted(months)
MONTH_MAP = {
    "Jan": "01", "January": "01",
    "Feb": "02", "February": "02",
    "Mar": "03", "March": "03",
    "Apr": "04", "April": "04",
    "May": "05",
    "Jun": "06", "June": "06",
    "Jul": "07", "July": "07",
    "Aug": "08", "August": "08",
    "Sep": "09", "Sept": "09", "September": "09",
    "Oct": "10", "October": "10",
    "Nov": "11", "November": "11",
    "Dec": "12", "December": "12",
}

def normalize_new_colname(col: str) -> str:
    col = col.strip()
    parts = col.split()
    if len(parts) < 3:
        return col

    # Last two tokens = month + year (e.g. "Apr 2025")
    month_raw = parts[-2].strip()
    year_raw  = parts[-1].strip()

    # Strip commas or punctuation from year (e.g. "2025-26" will fail intentionally)
    year = "".join(ch for ch in year_raw if ch.isdigit())[:4]
    month = month_raw[:3].title()   # "april" -> "Apr"

    if month not in MONTH_MAP or len(year) != 4:
        return col

    month_num = MONTH_MAP[month]
    metric = " ".join(parts[:-2]).strip()
    m = metric.lower().replace("%", "").strip()

    # Tonnage
    if "tonnage" in m:
        metric_clean = "Tonnage"
    # Trees saved
    elif "tree" in m:
        metric_clean = "Trees_Saved"
    # CO2 averted
    elif "co2" in m and ("avert" in m or "avoid" in m):
        metric_clean = "CO2_Kgs_Averted"
    # Households participating / participation
    elif ("household" in m or "hh" in m) and ("participation" in m or "participating" in m):
        metric_clean = "Households_Participating"
    # Segregation compliance %
    elif "segregation" in m or "compliance" in m:
        metric_clean = "Segregation_Compliance_Pct"
    else:
        metric_clean = metric.replace(" ", "_")

    return f"{metric_clean}_{year}-{month_num}"

@st.cache_data(show_spinner=False)
def load_and_prepare_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found.\nLooked for:\n- {CSV_WINDOWS_DEFAULT}\n- {CSV_LOCAL_FALLBACK}"
        )

    df = pd.read_csv(csv_path)
    # trim whitespace in headers
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    new_cols = {}
    for c in df.columns:
        new_cols[c] = normalize_new_colname(c)
    df.rename(columns=new_cols, inplace=True)
    # normalize Lat/Lon column names if they come as lowercase
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["lat", "latitude"]:
            colmap[c] = "Lat"
        elif cl in ["lon", "longitude", "long"]:
            colmap[c] = "Lon"

    if colmap:
        df.rename(columns=colmap, inplace=True)



    missing = [c for c in ID_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # numeric coercion for metric-month columns (now only 3 months, which is fine)
    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    if not metric_month_cols:
        raise ValueError("No metric-month columns like Impact_2024-04, Tonnage_2024-06 found.")

    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build long format
    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present, value_vars=metric_month_cols,
        var_name="Metric_Month", value_name="Value"
    )
    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df = long_df.drop(columns=["Metric_Month"]).sort_values(id_cols_present + ["Metric", "Date"])

    # cast key IDs to string
    for c in ["City", "Community", "Pincode"]:
        if c in df.columns:      df[c] = df[c].astype(str)
        if c in long_df.columns: long_df[c] = long_df[c].astype(str)

    # ensure Lat/Lon are numeric if present
    for c in ["Lat", "Lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, long_df, months, str(csv_path)

# ---------------- Sidebar (upload) ----------------
with st.sidebar:
    uploaded = st.file_uploader(
        "Upload dataset (CSV)",
        type=["csv"],
        help="Wide format: one row per community; monthly cols like Impact_2024-04, Tonnage_2024-06, ..."
    )
    st.caption("If no upload, the default CSV path above is used.")
    show_popup_charts = st.toggle(
        "Show charts in popups (slower)",
        value=False,
        help="Renders mini charts inside each popup. Turn off for fast map updates.",
        key="toggle_popup_charts"
    )

    st.caption("Map heatmap options:")

    heatmap_metric = st.selectbox(
        "Heatmap by",
        options=["None", "Participation Percent", "Dry Waste (kg)"],   # None = show default (circles); else show heatmap
        index=0,
        help="Show colored markers by Participation % or estimated Dry Waste (kg)",
        key="heatmap_metric"
    )


@st.cache_data(show_spinner=False)
def load_uploaded(file) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:

    df = pd.read_csv(file)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    # --- APPLY SAME NEW → OLD FORMAT CONVERSION ---
    new_cols = {}
    for c in df.columns:
        new_cols[c] = normalize_new_colname(c)
    df.rename(columns=new_cols, inplace=True)

    # --- Normalize Lat/Lon (Latitude/longitude included) ---
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["lat", "latitude", "Latitude".lower()]:
            colmap[c] = "Lat"
        elif cl in ["lon", "longitude", "Longitude".lower()]:
            colmap[c] = "Lon"
    if colmap:
        df.rename(columns=colmap, inplace=True)

    # Required IDs
    missing = [c for c in ID_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ID columns: {missing}")

    # Detect metric-month columns (converted format)
    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    if not metric_month_cols:
        raise ValueError("No metric–month columns found after normalization.")

    # Coerce numerics
    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build long format
    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present,
        value_vars=metric_month_cols,
        var_name="Metric_Month",
        value_name="Value",
    )

    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df = long_df.drop(columns=["Metric_Month"])

    # Cast IDs to string
    for c in ["City", "Community", "Pincode"]:
        if c in df.columns: df[c] = df[c].astype(str)
        if c in long_df.columns: long_df[c] = long_df[c].astype(str)

    # Ensure Lat/Lon numeric
    for c in ["Lat", "Lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, long_df, months, f"uploaded: {file.name}"



# ---------------- Initial load (defaults or upload) ----------------
try:
    if uploaded is not None:
        df_wide, df_long, months, data_src = load_uploaded(uploaded)
    else:
        df_wide, df_long, months, data_src = load_and_prepare_csv(CSV_DEFAULT)

    st.session_state["df_wide"]  = df_wide
    st.session_state["df_long"]  = df_long
    st.session_state["months"]   = months
    st.session_state["data_src"] = data_src
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ---------------- Minor UI theming fix (unchanged features) ----------------
st.markdown(
    """
    <style>
    div[data-baseweb="select"] { background-color: #FFFFFF !important; }
    div[data-baseweb="select"] span { color: white !important; }
    div[data-baseweb="select"] input { color: #36204D !important; }
    div[data-baseweb="menu"] { background-color: #FFFFFF !important; }
    div[data-baseweb="menu"] div[role="option"] { color: white !important; }
    div[data-baseweb="menu"] div[role="option"]:hover { background-color: #36204D22 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Data from session ----------------
df_wide  = st.session_state["df_wide"]
df_long  = st.session_state["df_long"]
months   = st.session_state["months"]
data_src = st.session_state["data_src"]


# Normalize key id columns to STRING (safety)
for col in ["Pincode", "Community", "City"]:
    if col in df_wide.columns:
        df_wide[col] = df_wide[col].astype(str)
    if col in df_long.columns:
        df_long[col] = df_long[col].astype(str)

# ---------------- Title ----------------
st.markdown("<h1>Smart Waste Analytics — FY 2024–25</h1>", unsafe_allow_html=True)
st.caption(f"Data source: {data_src}")

# ---------------- Global Filters ----------------
st.markdown("### 🔎 Global Filters")
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.2])

city_opts = sorted(df_wide["City"].dropna().unique().tolist()) if "City" in df_wide else []
comm_opts = sorted(df_wide["Community"].dropna().unique().tolist()) if "Community" in df_wide else []
pin_opts  = sorted(df_wide["Pincode"].dropna().unique().tolist()) if "Pincode" in df_wide else []

with c1:
    sel_city = st.multiselect("City", city_opts, placeholder="All", key="filter_city")
with c2:
    sel_comm = st.multiselect("Community", comm_opts, placeholder="All", key="filter_comm")
with c3:
    sel_pin  = st.multiselect("Pincode", pin_opts,  placeholder="All", key="filter_pin")
with c4:
    start_m, end_m = st.select_slider(
        "Date range (month)",
        options=months,
        value=(months[0], months[-1]),
        key="filter_month_range"
    )

def apply_filters(dfw, dfl):
    dfw = dfw.copy(); dfl = dfl.copy()
    for col in ["Pincode", "Community", "City"]:
        if col in dfw: dfw[col] = dfw[col].astype(str)
        if col in dfl: dfl[col] = dfl[col].astype(str)

    sel_city_s = [str(x) for x in sel_city] if sel_city else []
    sel_comm_s = [str(x) for x in sel_comm] if sel_comm else []
    sel_pin_s  = [str(x) for x in sel_pin]  if sel_pin  else []

    mask_w = pd.Series(True, index=dfw.index)
    if sel_city_s: mask_w &= dfw["City"].isin(sel_city_s)
    if sel_comm_s: mask_w &= dfw["Community"].isin(sel_comm_s)
    if sel_pin_s:  mask_w &= dfw["Pincode"].isin(sel_pin_s)
    dfw_f = dfw[mask_w].copy()

    mask_l = pd.Series(True, index=dfl.index)
    if sel_city_s: mask_l &= dfl["City"].isin(sel_city_s)
    if sel_comm_s: mask_l &= dfl["Community"].isin(sel_comm_s)
    if sel_pin_s:  mask_l &= dfl["Pincode"].isin(sel_pin_s)
    d0 = pd.to_datetime(start_m + "-01"); d1 = pd.to_datetime(end_m + "-01")
    mask_l &= (dfl["Date"] >= d0) & (dfl["Date"] <= d1)
    dfl_f = dfl[mask_l].copy()
    return dfw_f, dfl_f

dfw_filt, dfl_filt = apply_filters(df_wide, df_long)


dfw_filt = dfw_filt.copy()

selected_month = end_m  # This uses the month from your slider/filter (e.g. "2024-06")

seg_col = f"Segregation_Compliance_Pct_{selected_month}"
co2_col = f"CO2_Kgs_Averted_{selected_month}"

if seg_col in dfw_filt.columns and co2_col in dfw_filt.columns:
    dfw_filt["Participation_Pct"] = dfw_filt[seg_col]
    dfw_filt["Dry_Waste"] = dfw_filt[co2_col] / 2.18
else:
    dfw_filt["Participation_Pct"] = np.nan
    dfw_filt["Dry_Waste"] = np.nan



# ---------------- Summary KPIs ----------------
def kpi_value(dfl, metric, agg="sum"):
    s = dfl.loc[dfl["Metric"] == metric, "Value"]
    if s.empty: return 0.0
    return float(s.sum() if agg == "sum" else s.mean())

n_communities = dfw_filt["Community"].nunique() if "Community" in dfw_filt else 0
n_cities      = dfw_filt["City"].nunique()      if "City" in dfw_filt else 0
total_tonnage = kpi_value(dfl_filt, "Tonnage", "sum")
total_co2     = kpi_value(dfl_filt, "CO2_Kgs_Averted", "sum")
avg_comp      = kpi_value(dfl_filt, "Segregation_Compliance_Pct", "mean")
total_hh      = kpi_value(dfl_filt, "Households_Participating", "sum")



st.markdown("### 📊 Summary")
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
k1.metric("Communities", n_communities)
k2.metric("Cities", n_cities)
k3.metric("Total Tonnage", f"{total_tonnage:,.0f}")
k4.metric("CO₂ Averted (kg)", f"{total_co2:,.0f}")
k5.metric("Avg Segregation (%)", f"{avg_comp:,.1f}")
k6.metric("Active Households", f"{total_hh:,.0f}")
st.caption(f"Period: **{start_m} → {end_m}**")

# ---------------- Tabs ----------------
tab_map, tab_insights = st.tabs(["🗺️ 2D Map & Popups", "🧠 Insights"])

# --- Helpers for charts/popups ---
def _to_data_uri(fig, w=340):
    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=180)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' style='width:{w}px;height:auto;border:0;'/>"



def _distinct_colors(n):
    cmaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1]
    colors = []
    i = 0
    while len(colors) < n:
        cmap = cmaps[i % len(cmaps)]
        M = cmap.N
        take = min(n - len(colors), M)
        for j in range(take):
            colors.append(cmap(j / max(M - 1, 1)))
        i += 1
    return colors[:n]

@st.cache_data(show_spinner=False)
def summarize_for_popup(dfl_filtered: pd.DataFrame, community_id: str, pincode: str|None):
    d = dfl_filtered.copy()
    d["Community"] = d["Community"].astype(str)
    if pincode is not None and "Pincode" in d.columns:
        d["Pincode"] = d["Pincode"].astype(str)

    d = d[d["Community"] == str(community_id)]
    if pincode is not None:
        d = d[d["Pincode"] == str(pincode)]

    def agg(metric: str, how="sum") -> float:
        s = d.loc[d["Metric"] == metric, "Value"]
        if s.empty:
            return 0.0
        return float(s.sum() if how == "sum" else s.mean())

    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_kg = 0.0
    for m in dry_candidates:
        s = d.loc[d["Metric"] == m, "Value"]
        if not s.empty:
            dry_kg = float(s.sum())
            break

    co2 = dry_kg * CO2_PER_KG_DRY
    
    trees = agg("Trees_Saved", "sum")

    return {
        "tonnage":    agg("Tonnage", "sum"),
        "co2":        co2,
        "households": agg("Households_Participating", "sum"),
        "seg_pct":    agg("Segregation_Compliance_Pct", "mean"),
        "trees":      trees,
    }

@st.cache_data(show_spinner=False)
def monthly_series(df_long, community: str, metric: str):
    d = df_long[
        (df_long["Community"].astype(str) == str(community)) &
        (df_long["Metric"] == metric)
    ][["Date", "Value"]].sort_values("Date").copy()
    return d

@st.cache_data(show_spinner=False)
def popup_charts_for_comm(dfl_filtered: pd.DataFrame, community_id: str):
    BRAND = BRAND_PRIMARY
    dm = dfl_filtered.copy()
    dm["Community"] = dm["Community"].astype(str)
    dm = dm[dm["Community"] == str(community_id)]
    if dm.empty:
        return "", ""

    dm["MonthKey"] = dm["Date"].dt.to_period("M")

    # ------------------ TONNAGE (Plotly static PNG for popup) ------------------
    import plotly.express as px
    import plotly.io as pio

    bar_img = ""
    d_ton = dm[dm["Metric"] == "Tonnage"][["MonthKey", "Value"]].copy()
    if not d_ton.empty:
        d_ton["MonthLabel"] = [period.to_timestamp().strftime("%b") for period in d_ton["MonthKey"]]
        fig, ax = plt.subplots(figsize=(4.0, 1.4), dpi=120)
        ax.plot(d_ton["MonthLabel"], d_ton["Value"], marker="o", lw=1.6, color=BRAND)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
        ax.tick_params(axis="x", labelsize=8, colors=BRAND)
        ax.tick_params(axis="y", labelsize=8, colors=BRAND)
        ax.grid(alpha=0.12, axis="y")
        plt.xticks(rotation=45)
        bar_img = _to_data_uri(fig, w=380)


            

    # ------------------ CO2 DONUT (Plotly interactive preferred) ------------------
    # ------------------ CO2 DONUT (matplotlib with values) ------------------
    donut_img = ""
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_month = None
    for m in dry_candidates:
        cur = dm[dm["Metric"] == m][["MonthKey", "Value"]].copy()
        if not cur.empty:
            cur["Value"] = pd.to_numeric(cur["Value"], errors="coerce").fillna(0.0)
            dry_month = cur
            break




    if dry_month is not None:
        d = dry_month.groupby("MonthKey", as_index=False)["Value"].sum().sort_values("MonthKey")
        co2_vals = (d["Value"] * CO2_PER_KG_DRY).clip(lower=0.0).to_numpy()
        labels = [p.to_timestamp().strftime("%b") for p in d["MonthKey"]]
        colors = _distinct_colors(len(labels))

        # --- Donut chart with slice labels ---
        fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=120)
        wedges, texts, autotexts = ax.pie(
            co2_vals,
            labels=labels,  # ✅ show month beside slice
            autopct=lambda pct: (f'{pct:.1f}%') if pct > 0 else '',
            wedgeprops=dict(width=0.60, edgecolor='white', linewidth=1.2),
            startangle=90,
            colors=colors,
            pctdistance=0.7,
            labeldistance=1.05  # slight spacing between label and slice
        )
        ax.set(aspect="equal")

        # Format labels & percentages
        for t in texts:
            t.set_fontsize(8)
            t.set_color("#36204D")
            t.set_weight("bold")

        from matplotlib.colors import to_rgb
        for wedge, autotext in zip(wedges, autotexts):
            r, g, b = wedge.get_facecolor()[:3]
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            autotext.set_color('#ffffff' if lum < 0.6 else '#222222')
            autotext.set_fontsize(8)
            autotext.set_weight('bold')

        # Center text inside donut
        ax.text(0, 0, "CO₂\nAverted", ha="center", va="center",
                fontsize=10, color='purple', fontweight="bold")

        plt.tight_layout(pad=0.3)
        donut_img = _to_data_uri(fig, w=200)
        plt.close(fig)

        # --- Legend below the donut ---
        

        
        donut_img = f"<div style='text-align:center;'>{donut_img}</div>"
    # return HTML fragments (bar_img and donut_img)
    return bar_img, donut_img




def jitter_duplicates(df, lat_col="Lat", lon_col="Lon", jitter_deg=0.00025):
    """
    Move markers that share the same (Lat, Lon) into a tiny circle around
    the original location so each one gets a working tooltip/popup.
    jitter_deg ~ 0.00025 ≈ 25–30 meters.
    """
    df = df.copy()
    if lat_col not in df or lon_col not in df:
        return df
    gb = df.groupby([df[lat_col].round(6), df[lon_col].round(6)])
    for _, idx in gb.groups.items():
        n = len(idx)
        if n > 1:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = jitter_deg
            df.loc[idx, lat_col] = df.loc[idx, lat_col].to_numpy() + r * np.sin(angles)
            df.loc[idx, lon_col] = df.loc[idx, lon_col].to_numpy() + r * np.cos(angles)
    return df

import matplotlib
def get_heat_color(value, vmin, vmax):
    cmap = matplotlib.cm.get_cmap("RdYlGn")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(value))
    return matplotlib.colors.rgb2hex(rgba)





# ---------------- Map Tab ----------------
with tab_map:
    has_latlon = (
        "Lat" in dfw_filt.columns and
        "Lon" in dfw_filt.columns and
        dfw_filt[["Lat", "Lon"]].notna().all(axis=1).any()
    )

    if not has_latlon:
        st.warning("Map needs coordinates. Ensure **Lat/Lon** columns exist in the CSV.")
        st.info("Click markers to see details here (after coordinates are available).")
        selected_comm, selected_pin = None, None
    else:
        valid = dfw_filt.dropna(subset=["Lat", "Lon"])
        valid = jitter_duplicates(valid)

        if heatmap_metric == "Participation Percent":
            legend_html = """
            <div style='padding:12px 18px; background:#191b1f; border-radius:16px; width:305px; border:1.5px solid #DDD; box-shadow:1px 2px 12px #0003; margin-bottom:14px; margin-left:4px; color:#fff;'>
                <b style='font-size:17px; color:#fff;'>Participation % Scale</b><br/>
                <div style='display:flex; align-items:center; margin-top:6px;'>
                    <div style='background:#cc0002;width:35px;height:14px;'></div>
                    <div style='background:#DB680A;width:35px;height:14px;'></div>
                    <div style='background:#FFFF00;width:35px;height:14px;'></div>
                    <div style='background:#00FF00;width:35px;height:14px;'></div>
                    <span style='margin-left:14px;font-size:14px; color:#fff; font-weight:600;'>Low → High</span>
                </div>
                <span style='font-size:13px; color:#fff;'>Participation Intensity</span>
            </div>
            """
        elif heatmap_metric == "Dry Waste (kg)":
            legend_html = """
            <div style='padding:12px 18px; background:#191b1f; border-radius:16px; width:305px; border:1.5px solid #DDD; box-shadow:1px 2px 12px #0003; margin-bottom:14px; margin-left:4px; color:#fff;'>
                <b style='font-size:17px; color:#fff;'>Dry Waste Scale (kg)</b><br/>
                <div style='display:flex; align-items:center; margin-top:6px;'>
                    <div style='background:#00FF00;width:35px;height:14px;'></div>
                    <div style='background:#FFFF00;width:35px;height:14px;'></div>
                    <div style='background:#DB680A;width:35px;height:14px;'></div>
                    <div style='background:#cc0002;width:35px;height:14px;'></div>
                    <span style='margin-left:14px;font-size:14px; color:#fff; font-weight:600;'>Low → High</span>
                </div>
                <span style='font-size:13px; color:#fff;'>Dry Waste Intensity</span>
            </div>
            """
        else:
            legend_html = ""



        lat0 = float(valid["Lat"].mean())
        lon0 = float(valid["Lon"].mean())
        #fmap = folium.Map(location=[lat0, lon0], zoom_start=11, tiles="cartodbpositron")
        fmap = folium.Map(location=[lat0, lon0], zoom_start=11, tiles=None)

        # Add colorful HOT map
        folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
            name='OSM HOT',
            attr='OSM Hot'
        ).add_to(fmap)

        cluster = MarkerCluster().add_to(fmap)

        # ----------------- Cluster color override (purple) -----------------
        css = """
        <style>
        .marker-cluster-small {
        background-color: rgba(54,32,77,0.85) !important;   /* dark purple */
        color: #fff !important;
        border: 2px solid rgba(54,32,77,0.95) !important;
        }
        .marker-cluster-medium {
        background-color: rgba(115,77,155,0.85) !important; /* medium purple */
        color: #fff !important;
        border: 2px solid rgba(115,77,155,0.95) !important;
        }
        .marker-cluster-large {
        background-color: rgba(138,43,226,0.85) !important; /* brighter purple */
        color: #fff !important;
        border: 2px solid rgba(138,43,226,0.95) !important;
        }
        .marker-cluster div {
        box-shadow: none !important;
        font-weight: 700 !important;
        }
        </style>
        """

        class CssInject(MacroElement):
            def __init__(self, css_text):
                super().__init__()
                self._template = Template(css_text)
        from branca.element import Element
        
        fmap.get_root().html.add_child(CssInject(css))
        # ---- JS to flip popup above/below depending on marker position ----
        class JsInject(MacroElement):
            def __init__(self, js_text):
                super().__init__()
                self._template = Template(js_text)

        # hook into the correct map variable name
        map_name = fmap.get_name()
        js = f"""
        <script>
        (function() {{
        const map = {map_name};
        // Adjust popup offset based on marker's vertical screen position
        function adjustPopup(e) {{
            const popup = e.popup;
            const container = popup._container;
            if (!container) return;

            const mapH = map.getSize().y;
            const h = container.offsetHeight || 260;

            // y position of the marker in screen pixels
            const y = map.latLngToContainerPoint(popup.getLatLng()).y;

            // If marker is in lower half, show popup ABOVE (negative offset).
            // If marker is in upper half, show popup BELOW (positive offset).
            if (y > mapH / 2) {{
            popup.setOffset(L.point(0, -(h/2 + 20)));
            }} else {{
            popup.setOffset(L.point(0, +(h/2 + 20)));
            }}
            popup.update();

            // Nudge map so popup + nearby points stay in view
            // (works with Leaflet's autoPan to avoid hiding markers)
            setTimeout(function() {{
            try {{
                map.panInsideBounds(map.getBounds(), {{ paddingTopLeft:[12,12], paddingBottomRight:[12,12] }});
            }} catch(e) {{}}
            }}, 0);
        }}

        map.on('popupopen', adjustPopup);
        }})();
        </script>
        """

        fmap.get_root().html.add_child(JsInject(js))


        # --- Heatmap Coloring Logic ---
        if heatmap_metric == "Participation Percent":
            heat_column = "Participation_Pct"
        elif heatmap_metric == "Dry Waste (kg)":
            heat_column = "Dry_Waste"
        else:
            heat_column = None
        if heat_column:
            vmin = valid[heat_column].min()
            vmax = valid[heat_column].max()

        comm_arr = valid["Community"].astype(str).to_numpy()
        pin_arr  = valid["Pincode"].astype(str).to_numpy()
        lat_arr  = valid["Lat"].astype(float).to_numpy()
        lon_arr  = valid["Lon"].astype(float).to_numpy()
        city_arr = valid["City"].astype(str).to_numpy() if "City" in valid else np.array([""] * len(valid))

        # Precompute heat range if needed
        if heat_column:
            vmin = valid[heat_column].min()
            vmax = valid[heat_column].max()


                # ---------- ADD DENSITY HEATMAP LAYER (if requested) ----------
        # Build a list of [lat, lon, weight] where weight is the metric (Participation_Pct or Dry_Waste).
        # For Participation_Pct we use the percent value directly (0-100). For Dry_Waste (kg) we normalize
        # weights to the 0-1 range to avoid extremely large influence from outliers.
        if heat_column:
            heat_data = []
            # choose normalization for Dry_Waste so heatmap intensity stays reasonable
            if heat_column == "Dry_Waste":
                raw = valid[heat_column].fillna(0.0).astype(float)
                # avoid division by zero
                max_raw = raw.max() if raw.max() > 0 else 1.0
                norm_weights = (raw / max_raw).clip(0.0, 1.0)
                for r_lat, r_lon, w in zip(valid["Lat"], valid["Lon"], norm_weights):
                    heat_data.append([float(r_lat), float(r_lon), float(w)])
            else:
                # Participation_Pct — use value/100 so weights are 0..1
                raw = valid[heat_column].fillna(0.0).astype(float)
                for r_lat, r_lon, w in zip(valid["Lat"], valid["Lon"], raw / 100.0):
                    heat_data.append([float(r_lat), float(r_lon), float(w)])

            # Add HeatMap as a transparent overlay (radius and blur control density)
            # You can tweak radius and blur if you want stronger/weaker spread.
            


            if heat_column:
                heat_data = []

                if heat_column == "Dry_Waste":
                    raw = valid[heat_column].fillna(0.0).astype(float)
                    max_raw = raw.max() if raw.max() > 0 else 1.0
                    weights = (raw / max_raw).clip(0, 1)
                else:
                    raw = valid[heat_column].fillna(0.0).astype(float)
                    weights = (raw / 100.0).clip(0, 1)

                for r_lat, r_lon, w in zip(valid["Lat"], valid["Lon"], weights):
                    heat_data.append([float(r_lat), float(r_lon), float(w)])

                
                if heat_column == "Dry_Waste":
                    
                    gradient = {
                        0.0: "#00FF00",   # green = low
                        0.33: "#FFFF00",
                        0.66: "#DB680A",
                        1.0: "#cc0002",   # red = high
                    }
                else:
                    
                    gradient = {
                        0.0: "#cc0002",
                        0.33: "#DB680A",
                        0.66: "#FFFF00",
                        1.0: "#00FF00",
                    }

                HeatMap(
                    heat_data,
                    name=f"Density: {heatmap_metric}",
                    radius=25,
                    blur=15,
                    min_opacity=0.25,
                    max_zoom=12,
                    gradient=gradient,
                ).add_to(fmap)

                

        

        # Add a LayerControl so users can toggle heatmap on/off in the map UI
        folium.LayerControl(collapsed=True).add_to(fmap)


        # Add markers (do NOT call st_folium inside this loop)
        for i, (comm, pin, lat, lon, city) in enumerate(zip(comm_arr, pin_arr, lat_arr, lon_arr, city_arr)):
            stats = summarize_for_popup(dfl_filt, community_id=comm, pincode=pin)
            if (
                stats["tonnage"] == 0 and
                stats["co2"] == 0 and
                stats["households"] == 0 and
                stats["seg_pct"] == 0 and
                stats["trees"] == 0
            ):
                continue

            # optionally render popup charts (but do not make marker creation conditional on this)
            bar_img, donut_img = "", ""
            if show_popup_charts:
                try:
                    bar_img, donut_img = popup_charts_for_comm(dfl_filt, comm)
                except Exception:
                    bar_img, donut_img = "", ""

            
            community_id_val = valid.iloc[i]["community_id"] if "community_id" in valid.columns else ""

            # build popup HTML (always do this)
            popup_html = f"""
            <div style='font-family:Poppins; width:420px; padding:6px;'>

            <div style='display:flex; flex-direction:row; flex-wrap:wrap; justify-content:space-between; align-items:center;'>

                <!-- LEFT COLUMN -->
                <div style='flex:1.1; min-width:180px; max-width:230px; display:flex; flex-direction:column; align-items:flex-start;'>

                <!-- TOP: Community + Pincode + City -->
                <div style='width:100%; margin-bottom:8px;'>
                    <div style='display:flex; align-items:center; flex-wrap:wrap;'>
                        <div style='font-size:20px; font-weight:700; color:#36204D;'>{comm}</div>
                        <div style='font-size:15px; color:#555; margin-left:10px;'>({pin})</div>
                    </div>

                    <div style='font-size:13px; color:#777;'>Community ID: {community_id_val}</div>
                    <div style='font-size:13px; color:#777;'>City: {city}</div>
                </div>
                <!-- STATS -->
                <div style='font-size:15px; margin-bottom:5px; line-height:1.25;'>
                    <img src="{RECYCLE_ICON}" width="20" style="vertical-align:middle; margin-right:6px;">
                    <b style="font-size:16px; color:#36204D;">{stats['seg_pct']:.1f}% Segregation</b>
                </div>
                <div style='font-size:15px; margin-bottom:5px;'>
                    <img src="{HOUSE_ICON}" width="20" style="vertical-align:middle; margin-right:6px;">
                    <b style="font-size:16px; color:#36204D;">{stats['households']:,.0f} Households</b>
                </div>

                <!-- TREE SECTION -->
                <div style='text-align:center; width:100%; margin:12px 0 4px 0;'>
                    <img src="{TREE_ICON}" width="70" style="display:block; margin:0 auto 6px;">
                    <div style='font-size:18px; font-weight:700; color:#36204D;'>{stats['trees']:,.1f} Trees Saved</div>
                </div>
                </div>

                <!-- RIGHT COLUMN: Donut Chart -->
                <div style='flex:1; min-width:210px; max-width:250px; display:flex; justify-content:center; align-items:center; margin-left:-10px;'>
                {donut_img}
                </div>
            </div>

            <hr style='margin:12px 0 10px 0; width:98%; border:0.8px solid #ddd;'>

            <!-- BOTTOM: Tonnage Line Chart -->
            <div style='width:99%; margin:0 auto; text-align:left;'>
                <div style='font-size:16px; margin-bottom:5px; color:#36204D; font-weight:700;'>KGs</div>
                {bar_img}
            </div>
            </div>
            """




            # HEATMAP color for this marker
            if heat_column:
                val = valid.iloc[i][heat_column]
                color = get_heat_color(val, vmin, vmax)
            else:
                color = BRAND_PRIMARY

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="#333333",
                fill=True,
                fill_color="#333333",
                fill_opacity=0.85,
                tooltip=folium.Tooltip(f"{comm} • {pin}"),
                popup=folium.Popup(popup_html, max_width=380),
            ).add_to(cluster)


        if heatmap_metric != "None":
            st.markdown(legend_html, unsafe_allow_html=True)    

        # After all markers added, render the map once
        st.markdown("##### Map")
        map_event = st_folium(
            fmap,
            height=ST_MAP_HEIGHT,
            use_container_width=True,
            returned_objects=ST_RETURNED_OBJECTS,
            key="main_leaflet_map_v1"
        )


      
  

        selected_comm, selected_pin = None, None
        if map_event and map_event.get("last_object_clicked_tooltip"):
            tip = map_event["last_object_clicked_tooltip"]  # "COMMUNITY • PINCODE"
            parts = [p.strip() for p in tip.split("•")]
            if len(parts) == 2:
                selected_comm, selected_pin = parts[0], parts[1]
            else:
                selected_comm = parts[0]
        elif not dfw_filt.empty:
            selected_comm = str(dfw_filt.iloc[0]["Community"])
            selected_pin  = str(dfw_filt.iloc[0]["Pincode"])





    # --- KPIs for selected community ---
    cA, cB, cC, cD = st.columns(4)
    stats = summarize_for_popup(dfl_filt, community_id=selected_comm, pincode=selected_pin)
    cA.metric("Tonnage (kg)", f"{stats['tonnage']:,.0f}")
    cB.metric("CO₂ Averted (kg)", f"{stats['co2']:,.0f}")
    cC.metric("Households", f"{stats['households']:,.0f}")
    cD.metric("Segregation % (avg)", f"{stats['seg_pct']:.1f}")

    # --- Trends for selected community ---
    st.markdown("#### Trends (Selected Community)")

    ton = monthly_series(dfl_filt, selected_comm, "Tonnage")
    if not ton.empty:
        fig_ton = px.line(
            ton, x="Date", y="Value",
            title="Tonnage over Time",
            labels={"Value": "Tonnage (kg)", "Date": "Date"},
            markers=True,
        )
        fig_ton.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                              marker=dict(color=BRAND_PRIMARY))
        fig_ton.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(title=dict(text="Date", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            yaxis=dict(title=dict(text="Tonnage (kg)", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
        )
        st.plotly_chart(fig_ton, use_container_width=True)
    else:
        st.info("No tonnage data in this date range for the selected community.")

    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry = None
    for m in dry_candidates:
        s = monthly_series(dfl_filt, selected_comm, m)
        if not s.empty:
            dry = s
            break

    if dry is not None and not dry.empty:
        co2 = dry.copy()
        co2["CO2_kg"] = (co2["Value"] * CO2_PER_KG_DRY).clip(lower=0.0)
        fig_co2 = px.line(
            co2, x="Date", y="CO2_kg",
            markers=True,
            title="CO₂ Averted (Calculated) over Time",
            labels={"CO2_kg": "CO₂ Averted (kg)", "Date": "Date"},
        )
        fig_co2.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                              marker=dict(color=BRAND_PRIMARY, size=6))
        fig_co2.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(title=dict(text="Date", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            yaxis=dict(title=dict(text="CO₂ Averted (kg)", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            showlegend=False,
        )
        st.plotly_chart(fig_co2, use_container_width=True)
    else:
        st.info("No dry/tonnage series available to compute CO₂ for this community.")

    seg = monthly_series(dfl_filt, selected_comm, "Segregation_Compliance_Pct")
    if not seg.empty:
        fig_seg = px.line(
            seg, x="Date", y="Value",
            markers=True,
            title="Segregation % over Time",
            labels={"Value": "Segregation (%)", "Date": "Date"},
        )
        fig_seg.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                              marker=dict(color=BRAND_PRIMARY, size=6))
        fig_seg.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(title=dict(text="Date", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            yaxis=dict(title=dict(text="Segregation (%)", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            showlegend=False,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

# ---------------- Insights Tab ----------------
with tab_insights:
    st.markdown("### 🧠 Auto Insights (All Cities in Selected Date Range)")

    d0 = pd.to_datetime(start_m + "-01")
    d1 = pd.to_datetime(end_m + "-01")
    dfl_date = df_long[(df_long["Date"] >= d0) & (df_long["Date"] <= d1)].copy()

    for col in ["City", "Community", "Pincode"]:
        if col in dfl_date.columns:
            dfl_date[col] = dfl_date[col].astype(str)


    def _brand_axes(fig, title=None):
        fig.update_layout(
            title=title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else None),
            font=dict(family="Poppins", color=BRAND_PRIMARY, size=14),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            legend=dict(font=dict(color="#36204D", size=13)),  # <-- Added: sets legend text color
        )
        fig.update_xaxes(title_font=dict(color=BRAND_PRIMARY, size=12), tickfont=dict(color=BRAND_PRIMARY, size=11),
                        gridcolor="#EEE", zerolinecolor="#EEE")
        fig.update_yaxes(title_font=dict(color=BRAND_PRIMARY, size=12), tickfont=dict(color=BRAND_PRIMARY, size=11),
                        gridcolor="#EEE", zerolinecolor="#EEE")
        return fig


    
    sum_metrics  = ["Tonnage", "CO2_Kgs_Averted", "Households_Participating"]
    mean_metrics = ["Segregation_Compliance_Pct"]

    city_sum = (
        dfl_date[dfl_date["Metric"].isin(sum_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    city_mean = (
        dfl_date[dfl_date["Metric"].isin(mean_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value", aggfunc="mean", fill_value=0.0)
        .reset_index()
    )

    def _top_city(df, metric, how="max"):
        if df.empty or metric not in df.columns:
            return "—", 0.0
        row = df.loc[df[metric].idxmax()] if how == "max" else df.loc[df[metric].idxmin()]
        return str(row["City"]), float(row[metric])

    colA, colB, colC, colD = st.columns(4)
    t_city, t_val = _top_city(city_sum, "Tonnage", "max")
    c_city, c_val = _top_city(city_sum, "CO2_Kgs_Averted", "max")
    h_city, h_val = _top_city(city_sum, "Households_Participating", "max")
    s_city, s_val = _top_city(city_mean, "Segregation_Compliance_Pct", "max")

    with colA:
        st.caption("Top city by Tonnage")
        st.subheader(t_city)
        st.success(f"↑ {t_val:,.0f}")
    with colB:
        st.caption("Top city by CO₂ averted (kg)")
        st.subheader(c_city)
        st.success(f"↑ {c_val:,.0f}")
    with colC:
        st.caption("Top city by Households")
        st.subheader(h_city)
        st.success(f"↑ {h_val:,.0f}")
    with colD:
        st.caption("Highest Avg Segregation (%)")
        st.subheader(s_city)
        st.success(f"↑ {s_val:,.1f}%")

    st.markdown("---")

    if not city_sum.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                city_sum.sort_values("Tonnage", ascending=False),
                x="City", y="Tonnage",
                text="Tonnage", title="Total Tonnage by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c2:
            fig = px.bar(
                city_sum.sort_values("CO2_Kgs_Averted", ascending=False),
                x="City", y="CO2_Kgs_Averted",
                text="CO2_Kgs_Averted", title="CO₂ Averted (kg) by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                city_sum.sort_values("Households_Participating", ascending=False),
                x="City", y="Households_Participating",
                text="Households_Participating", title="Households by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)


        with c4:
            # FIX: safely sort only if the column exists
            if "Segregation_Compliance_Pct" in city_mean.columns:
                df_sorted = city_mean.sort_values("Segregation_Compliance_Pct", ascending=False)
            else:
                df_sorted = city_mean  # avoid crash

            fig = px.bar(
                df_sorted,
                x="City",
                y="Segregation_Compliance_Pct" if "Segregation_Compliance_Pct" in df_sorted.columns else None,
                text="Segregation_Compliance_Pct" if "Segregation_Compliance_Pct" in df_sorted.columns else None,
                title="Avg Segregation (%) by City",
            )

            fig.update_traces(
                marker_color=BRAND_PRIMARY,
                texttemplate="%{text:.1f}%" if "Segregation_Compliance_Pct" in df_sorted.columns else "",
                textposition="outside"
            )

            st.plotly_chart(_brand_axes(fig), use_container_width=True)
            

    st.markdown("---")

    topN = 10
    comm_tonn = (
        dfl_date[dfl_date["Metric"] == "Tonnage"]
        .groupby(["City", "Community", "Pincode"], as_index=False)["Value"].sum()
        .rename(columns={"Value": "Tonnage"})
        .sort_values("Tonnage", ascending=False)
        .head(topN)
    )

    st.markdown(f"#### Top {topN} Communities by Tonnage (All Cities)")
    if not comm_tonn.empty:
        fig_comm = px.bar(
            comm_tonn,
            x="Community", y="Tonnage",
            color="City",
            title="Top Communities by Total Tonnage",
            labels={"Value": "Tonnage", "Community": "Community"},
        )
        fig_comm.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        fig_comm = _brand_axes(fig_comm)
        st.plotly_chart(fig_comm, use_container_width=True)
        st.caption("Tip: Hover a bar to see its city and pincode.")
    else:
        st.info("No community tonnage available in this date range.")

    st.markdown("---")

    st.write("The table below reflects the **current filters** (city/community/pincode + date range).")
    filtered_csv = dfl_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Trends (filtered CSV)",
        data=filtered_csv,
        file_name="trends_filtered.csv",
        mime="text/csv",
        key="dl_trends_bottom",
    )
    st.dataframe(dfl_filt, use_container_width=True, height=420)


