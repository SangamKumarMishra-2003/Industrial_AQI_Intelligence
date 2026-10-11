# Industrial AQI Intelligence

A Streamlit machine-learning dashboard built around `air_quality_assessment.csv`.

## Features
- Trains and compares Random Forest, Extra Trees, and HistGradientBoosting regression models.
- Uses timestamp-sorted 80/20 holdout validation.
- Manual pollutant/weather/industrial input form with AQI prediction.
- Hourly what-if outlook based on typical hour-level feature medians.
- Dataset upload, explorer, model metrics, and CSV export.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Required CSV columns
`Date`, `Time`, `PM2_5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `Temperature`, `Humidity`, `Wind_Speed`, `Wind_Direction`, `Pressure`, `Rainfall`, `Traffic_Index`, `Industrial_Activity`, `AQI`

## Accuracy note
On the provided dataset, a preliminary chronological 80/20 holdout benchmark gave Random Forest MAE ≈ 9.94, RMSE ≈ 12.41, R² ≈ 0.755. App settings/model versions can shift these values slightly. This is an estimate on the supplied data, not a guarantee of real-world accuracy. The hourly view is a scenario simulation using typical historical hourly conditions, not a causal forecast. Dataset AQI values are treated as the target label and are not independently verified against official AQI calculation rules.
