import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "air_quality_assessment.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

st.set_page_config(page_title="Industrial AQI Intelligence", page_icon="🌫️", layout="wide")

POLLUTANTS = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3"]
WEATHER = ["Temperature", "Humidity", "Wind_Speed", "Wind_Direction", "Pressure", "Rainfall"]
OPERATIONS = ["Traffic_Index", "Industrial_Activity"]
NUMERIC_INPUTS = POLLUTANTS + WEATHER + OPERATIONS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; color: #132238; }
.block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
.hero { background: linear-gradient(120deg,#102a43 0%,#174b68 55%,#087e8b 100%); padding: 30px 34px; border-radius: 22px; color: white; margin-bottom: 22px; box-shadow: 0 12px 32px rgba(16,42,67,.16); }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size: 2.2rem; letter-spacing:-1px; margin:0 0 8px 0; color:white; }
.hero p { color:#d8e8f2; margin:0; font-size:1rem; }
.kpi { background:#fff; border:1px solid #e3eaf2; border-radius:16px; padding:18px 20px; min-height:112px; box-shadow:0 4px 15px rgba(22,43,66,.035); }
.kpi-label { color:#62758a; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.kpi-value { font-family:'Space Grotesk',sans-serif; color:#102a43; font-size:1.9rem; font-weight:700; margin-top:7px; }
.kpi-sub { color:#718096; font-size:.8rem; margin-top:2px; }
.section-title {font-family:'Space Grotesk',sans-serif; font-size:1.22rem; font-weight:700; color:#102a43; margin:8px 0 12px;}
.panel { background:white; border:1px solid #e3eaf2; border-radius:16px; padding:20px; }
.small-note { color:#63758a; font-size:.84rem; }
div[data-testid="stMetric"] { background:white; border:1px solid #e3eaf2; padding:16px; border-radius:14px; }
.stButton button[kind="primary"] {
    color: white;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #132238;
    background: white;
}

.stApp [data-testid="stMarkdownContainer"],
.stApp label {
    color: #132238;
}
</style>
""", unsafe_allow_html=True)

def category(aqi):
    if aqi <= 50: return "Good", "#15966a", "Air quality is considered good."
    if aqi <= 100: return "Satisfactory", "#70a83b", "Air quality is acceptable; unusually sensitive people may need caution."
    if aqi <= 200: return "Moderate", "#d99a16", "Sensitive people may experience discomfort; reduce prolonged heavy outdoor exertion if affected."
    if aqi <= 300: return "Poor", "#e46a35", "Limit prolonged outdoor exertion, especially for sensitive groups."
    if aqi <= 400: return "Very Poor", "#b83d55", "Reduce outdoor exposure and follow local health advisories."
    return "Severe", "#762c83", "Avoid unnecessary outdoor exertion and follow local health advisories."

def prepare_data(raw):
    raw = raw.copy()
    required = ["Date", "Time", *NUMERIC_INPUTS, "AQI"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    raw["DateTime"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), errors="coerce")
    for c in NUMERIC_INPUTS + ["AQI"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["DateTime", "AQI"]).sort_values("DateTime")
    raw["hour"] = raw["DateTime"].dt.hour
    raw["dayofweek"] = raw["DateTime"].dt.dayofweek
    raw["month"] = raw["DateTime"].dt.month
    raw["dayofyear"] = raw["DateTime"].dt.dayofyear
    raw["hour_sin"] = np.sin(2 * np.pi * raw["hour"] / 24)
    raw["hour_cos"] = np.cos(2 * np.pi * raw["hour"] / 24)
    raw["doy_sin"] = np.sin(2 * np.pi * raw["dayofyear"] / 365.25)
    raw["doy_cos"] = np.cos(2 * np.pi * raw["dayofyear"] / 365.25)
    features = NUMERIC_INPUTS + ["hour", "dayofweek", "month", "dayofyear", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return raw, features

@st.cache_data(show_spinner=False)
def load_default():
    return pd.read_csv(DATA_PATH)

@st.cache_resource(show_spinner="Training and validating models on your dataset…")
def train_models(csv_bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    data, features = prepare_data(raw)
    X, y = data[features], data["AQI"]
    if len(data) < 30:
        raise ValueError("At least 30 valid rows are recommended for training and time-ordered validation.")
    split = max(1, int(len(data) * 0.8))
    if split >= len(data): split = len(data)-1
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=260, min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=260, min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=-1),
        "Hist. Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, l2_regularization=1.0, random_state=42),
    }
    metrics, trained = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R²": r2_score(y_test, pred)})
        trained[name] = pipe
    metrics_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    # Refit selected model on all rows after unbiased time-ordered evaluation.
    best = candidates[best_name]
    final_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", best)])
    final_pipe.fit(X, y)
    return data, features, metrics_df, best_name, final_pipe

def make_row(values, timestamp):
    ts = pd.Timestamp(timestamp)
    row = dict(values)
    row.update({
        "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month, "dayofyear": ts.dayofyear,
        "hour_sin": np.sin(2*np.pi*ts.hour/24), "hour_cos": np.cos(2*np.pi*ts.hour/24),
        "doy_sin": np.sin(2*np.pi*ts.dayofyear/365.25), "doy_cos": np.cos(2*np.pi*ts.dayofyear/365.25)
    })
    return row

def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

raw_default = load_default()
with st.sidebar:
    st.markdown("## 🌫️ AQI Intelligence")
    st.caption("Industrial air quality • ML workspace")
    uploaded = st.file_uploader("Use another CSV dataset", type=["csv"], help="The CSV must include Date, Time, pollutant/weather/industrial columns, and AQI.")
    st.markdown("---")
    st.markdown("**Dataset features required**")
    st.caption("Date, Time, PM2_5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind_Speed, Wind_Direction, Pressure, Rainfall, Traffic_Index, Industrial_Activity, AQI")
    st.markdown("---")
    st.caption("Model evaluation uses a chronological 80/20 holdout. Predictions are estimates, not certified monitoring readings.")

csv_bytes = uploaded.getvalue() if uploaded else DATA_PATH.read_bytes()
try:
    data, features, metrics, best_name, model = train_models(csv_bytes)
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

latest = data.iloc[-1]
mae = float(metrics.iloc[0]["MAE"])
rmse = float(metrics.iloc[0]["RMSE"])
r2 = float(metrics.iloc[0]["R²"])
st.markdown(f"""
<div class="hero">
  <h1>Industrial AQI Intelligence</h1>
  <p>Predict, explore and simulate air quality across industrial time windows — powered by your dataset.</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "🧪 Predict AQI", "🕒 Hourly outlook", "🧠 Model performance", "📁 Dataset explorer"])

with tabs[0]:
    st.markdown('<div class="section-title">Model & dataset at a glance</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    with a: kpi("Records", f"{len(data):,}", f"{len(NUMERIC_INPUTS)} input variables")
    with b: kpi("Mean recorded AQI", f"{data.AQI.mean():.1f}", f"Range {data.AQI.min():.1f}–{data.AQI.max():.1f}")
    with c: kpi("Validation MAE", f"{mae:.2f}", "AQI points · lower is better")
    with d: kpi("Best model", best_name, f"Holdout R² {r2:.3f}")
    st.write("")
    left, right = st.columns([1.3,1])
    with left:
        st.markdown('<div class="section-title">AQI over time</div>', unsafe_allow_html=True)
        plotdata = data[["DateTime","AQI"]].copy()
        plotdata["Rolling mean (24 samples)"] = plotdata["AQI"].rolling(24, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plotdata.DateTime, y=plotdata.AQI, mode="lines", name="Recorded AQI", line=dict(color="#8bb6c9",width=1)))
        fig.add_trace(go.Scatter(x=plotdata.DateTime, y=plotdata["Rolling mean (24 samples)"], mode="lines", name="24-sample rolling mean", line=dict(color="#087e8b",width=3)))
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h",y=1.12), xaxis_title="", yaxis_title="AQI")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">AQI category distribution</div>', unsafe_allow_html=True)
        catdf = data[["AQI"]].copy()
        catdf["Category"] = catdf["AQI"].map(lambda x: category(x)[0])
        order = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
        counts = catdf.Category.value_counts().reindex(order,fill_value=0).reset_index()
        counts.columns=["Category","Records"]
        fig2 = px.bar(counts,x="Category",y="Records",color="Category",category_orders={"Category":order},
                      color_discrete_map={"Good":"#15966a","Satisfactory":"#70a83b","Moderate":"#d99a16","Poor":"#e46a35","Very Poor":"#b83d55","Severe":"#762c83"})
        fig2.update_layout(height=340,showlegend=False,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="",yaxis_title="Records")
        st.plotly_chart(fig2,use_container_width=True)
    st.info("AQI category labels use common Indian AQI bands as an indicative display. The dataset's AQI target is learned as provided; it is not recalculated from CPCB pollutant formulas.")

with tabs[1]:
    st.markdown('<div class="section-title">Enter sensor readings</div>', unsafe_allow_html=True)
    st.markdown("Adjust the values to simulate a plant/industrial site. Missing fields in training data are imputed with training medians.")
    with st.form("prediction_form"):
        p1,p2,p3 = st.columns(3)
        defaults = data[NUMERIC_INPUTS].median(numeric_only=True).to_dict()
        values = {}
        ranges = {
            "PM2_5":(0,700,0.1),"PM10":(0,800,0.1),"NO2":(0,400,0.1),"SO2":(0,100,0.1),"CO":(0,20,0.01),"O3":(0,150,0.1),
            "Temperature":(-10,55,0.1),"Humidity":(0,100,0.1),"Wind_Speed":(0,30,0.1),"Wind_Direction":(0,360,1),
            "Pressure":(970,1050,0.1),"Rainfall":(0,30,0.1),"Traffic_Index":(0,100,0.1),"Industrial_Activity":(0,100,0.1)
        }
        groups=[("Pollutants",POLLUTANTS),("Weather & environment",WEATHER),("Site activity",OPERATIONS)]
        cols=[p1,p2,p3]
        for col,(group,fields) in zip(cols,groups):
            with col:
                st.markdown(f"**{group}**")
                for f in fields:
                    lo,hi,step=ranges[f]
                    val=float(np.clip(defaults[f],lo,hi))
                    values[f]=st.number_input(f.replace("_"," "),min_value=float(lo),max_value=float(hi),value=val,step=float(step),key="manual_"+f)
        d1,d2=st.columns(2)
        with d1: chosen_date=st.date_input("Prediction date",value=pd.Timestamp(latest.DateTime).date())
        with d2: chosen_time=st.selectbox("Prediction time", options=[f"{h:02d}:00" for h in range(24)], index=int(pd.Timestamp(latest.DateTime).hour))
        submitted=st.form_submit_button("Predict AQI",type="primary",use_container_width=True)
    if submitted:
        ts=pd.Timestamp(f"{chosen_date} {chosen_time}")
        row=make_row(values,ts)
        pred=float(model.predict(pd.DataFrame([row])[features])[0])
        label,color,advice=category(pred)
        st.markdown("---")
        c1,c2=st.columns([1,2])
        with c1:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Predicted AQI</div><div class="kpi-value" style="color:{color}">{pred:.1f}</div><div class="kpi-sub">{label} · {ts.strftime("%d %b %Y, %H:%M")}</div></div>',unsafe_allow_html=True)
        with c2:
            st.markdown(f"### {label} air quality")
            st.write(advice)
            st.caption(f"Model: {best_name}. Validation MAE: {mae:.2f} AQI points; individual predictions can be less accurate.")
        gauge=go.Figure(go.Indicator(mode="gauge+number",value=pred,number={"font":{"size":34}},gauge={"axis":{"range":[0,max(300, float(data.AQI.quantile(.99)))]},"bar":{"color":color},"steps":[{"range":[0,50],"color":"#e2f5ed"},{"range":[50,100],"color":"#edf5dc"},{"range":[100,200],"color":"#fff0c8"},{"range":[200,300],"color":"#fde1d5"}]}))
        gauge.update_layout(height=220,margin=dict(l=25,r=25,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge,use_container_width=True)
        st.download_button("Download this prediction as CSV",pd.DataFrame([{**values,"DateTime":str(ts),"Predicted_AQI":round(pred,2),"Category":label,"Model":best_name}]).to_csv(index=False).encode(),file_name="aqi_prediction.csv",mime="text/csv")

with tabs[2]:
    st.markdown('<div class="section-title">24-hour outlook / what-if simulation</div>', unsafe_allow_html=True)
    st.warning("This is a scenario simulation, not a guaranteed future forecast. It uses typical values from each hour in your dataset, then predicts AQI. It assumes the historical hour-level profile is a reasonable proxy for the selected date.")
    target_date=st.date_input("Choose date for hourly outlook",value=pd.Timestamp(latest.DateTime).date(),key="outlook_date")
    hourly=[]
    # Use median historical conditions per hour, so each row represents a typical hour profile.
    hour_profiles=data.groupby("hour")[NUMERIC_INPUTS].median(numeric_only=True)
    for hour in range(24):
        vals=hour_profiles.loc[hour].to_dict() if hour in hour_profiles.index else data[NUMERIC_INPUTS].median().to_dict()
        ts=pd.Timestamp(f"{target_date} {hour:02d}:00")
        row=make_row(vals,ts)
        pred=float(model.predict(pd.DataFrame([row])[features])[0])
        hourly.append({"Time":ts,"Hour":f"{hour:02d}:00","Predicted AQI":pred,"Category":category(pred)[0]})
    hourly_df=pd.DataFrame(hourly)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hourly_df["Hour"],y=hourly_df["Predicted AQI"],mode="lines+markers",line=dict(color="#087e8b",width=3),marker=dict(size=7),fill="tozeroy",fillcolor="rgba(8,126,139,.10)",name="Predicted AQI"))
    fig.update_layout(height=380,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="Hour",yaxis_title="Estimated AQI",xaxis=dict(tickmode="linear",dtick=2))
    st.plotly_chart(fig,use_container_width=True)
    peak=hourly_df.loc[hourly_df["Predicted AQI"].idxmax()]
    low=hourly_df.loc[hourly_df["Predicted AQI"].idxmin()]
    a,b,c=st.columns(3)
    with a: kpi("Highest estimated hour",peak["Hour"],f"AQI {peak['Predicted AQI']:.1f}")
    with b: kpi("Lowest estimated hour",low["Hour"],f"AQI {low['Predicted AQI']:.1f}")
    with c: kpi("Daily simulated average",f"{hourly_df['Predicted AQI'].mean():.1f}","Based on typical hourly conditions")
    st.dataframe(hourly_df[["Hour","Predicted AQI","Category"]].round({"Predicted AQI":1}),use_container_width=True,hide_index=True)
    st.download_button("Export hourly outlook CSV",hourly_df[["Hour","Predicted AQI","Category"]].to_csv(index=False).encode(),file_name=f"hourly_aqi_outlook_{target_date}.csv",mime="text/csv")

with tabs[3]:
    st.markdown('<div class="section-title">Time-ordered model validation</div>', unsafe_allow_html=True)
    st.write("The earliest 80% of timestamp-sorted records are used for training; the latest 20% are held out for evaluation. This is a more realistic check than randomly mixing future and past rows.")
    m1,m2,m3=st.columns(3)
    with m1: st.metric("Best validation MAE",f"{mae:.2f} AQI points")
    with m2: st.metric("Best validation RMSE",f"{rmse:.2f} AQI points")
    with m3: st.metric("Best validation R²",f"{r2:.3f}")
    formatted_metrics = metrics.copy()

    for column, fmt in {
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.3f}"
    }.items():
        if column in formatted_metrics.columns:
            formatted_metrics[column] = formatted_metrics[column].map(
                lambda value: fmt.format(value) if pd.notna(value) else "N/A"
            )

    st.dataframe(
        formatted_metrics,
        use_container_width=True,
        hide_index=True
    )
    st.caption("MAE and RMSE are in AQI points. R² describes variance explained on the held-out period. None guarantees the same performance at a new plant or under conditions outside the dataset.")
    st.markdown("**How to improve reliability**")
    st.markdown("- Collect more continuous measurements across seasons and industrial operating states.\n- Verify sensor calibration and pollutant units; inspect outliers before retraining.\n- For genuine future forecasts, use lagged pollutant/AQI measurements and validate with rolling time windows.\n- Keep a separate plant/site test set before deploying the model operationally.")

with tabs[4]:
    st.markdown('<div class="section-title">Dataset explorer</div>', unsafe_allow_html=True)
    st.write(f"Rows after timestamp/AQI validation: **{len(data):,}** · Date range: **{data.DateTime.min()}** to **{data.DateTime.max()}**")
    n_missing=int(data[NUMERIC_INPUTS].isna().sum().sum())
    x1,x2,x3=st.columns(3)
    with x1: kpi("Missing feature cells",f"{n_missing:,}","Imputed during model fitting")
    with x2: kpi("Input variables",str(len(NUMERIC_INPUTS)),"Pollutants, weather & activity")
    with x3: kpi("Unique timestamps",f"{data.DateTime.nunique():,}","Duplicate timestamps may exist")
    selected_cols=st.multiselect("Columns to display",options=["DateTime","AQI"]+NUMERIC_INPUTS,default=["DateTime","AQI","PM2_5","PM10","Industrial_Activity"])
    st.dataframe(data[selected_cols].tail(500),use_container_width=True,hide_index=True)
    st.download_button("Download cleaned dataset",data[["DateTime"]+NUMERIC_INPUTS+["AQI"]].to_csv(index=False).encode(),file_name="cleaned_air_quality_data.csv",mime="text/csv")

st.markdown("---")
st.caption("Industrial AQI Intelligence · Research/demo tool. Do not use predictions as a substitute for certified air-quality instruments, statutory reporting, or local public-health guidance.")
