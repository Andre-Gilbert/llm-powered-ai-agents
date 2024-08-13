"""Forecasting tool."""

import os
from datetime import datetime, timedelta
from urllib import parse

import catboost as cb
import pandas as pd

_LAGS = 7
_MEDIAN_LATITUDE = {
    "Alaska": 61.1224,
    "Aleutian Islands": 51.7672,
    "Baja California": 32.36241665,
    "California": 36.6798333,
    "Chile": -32.443,
    "Greece": 38.07,
    "Hawaii": 19.3948326,
    "Idaho": 44.3036667,
    "Indonesia": -1.9559,
    "Italy": 44.204,
    "Japan": 36.306,
    "Mexico": 16.629,
    "Montana": 44.7548333,
    "Nevada": 38.158,
    "Oklahoma": 36.1025,
    "Oregon": 43.88341665,
    "Papua New Guinea": -5.604,
    "Philippines": 9.242,
    "Puerto Rico": 17.9971,
    "Russia": 50.895,
    "Tonga": -18.9217,
    "Turkey": 39.117,
    "Utah": 39.4306667,
    "Washington": 46.5871667,
    "Wyoming": 4,
}
_MEDIAN_LONGITUDE = {
    "Alaska": -151.1221,
    "Aleutian Islands": 178.314,
    "Baja California": -115.57125,
    "California": -118.8568333,
    "Chile": -71.303,
    "Greece": 22.52,
    "Hawaii": -155.2838333,
    "Idaho": -114.5975,
    "Indonesia": 122.566,
    "Italy": 10.717,
    "Japan": 141.044,
    "Mexico": -98.208,
    "Montana": -111.0218333,
    "Nevada": -117.8719,
    "Oklahoma": -97.5726,
    "Oregon": -121.93625,
    "Papua New Guinea": 151.269,
    "Philippines": 125.822,
    "Puerto Rico": -66.8571,
    "Russia": 150.9206,
    "Tonga": -174.5247,
    "Turkey": 28.975,
    "Utah": -111.4066667,
    "Washington": -122.1856667,
    "Wyoming": -110.7056667,
}


def get_recent_earthquakes(
    start_time: datetime = (datetime.now() - timedelta(days=30)).date(),
    end_time: datetime = datetime.now().date(),
    limit: int = 20000,
    min_depth: int = -100,
    max_depth: int = 1000,
    min_magnitude: int | None = None,
    max_magnitude: int | None = None,
    alert_level: str | None = None,
) -> pd.DataFrame:
    params = {
        "format": "csv",
        "starttime": start_time,
        "endtime": end_time,
        "limit": limit,
        "mindepth": min_depth,
        "maxdepth": max_depth,
        "eventtype": "earthquake",
    }
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude
    if alert_level is not None:
        params["alertlevel"] = alert_level
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + parse.urlencode(params)
    return pd.read_csv(url)


def get_regions() -> list[str]:
    df = get_recent_earthquakes()
    df["region"] = df.place.str.split(", ", expand=True)[1]
    df.region = df.region.fillna(df.place)
    df.region = df.region.replace({"CA": "California", "B.C.": "Baja California"})
    return set(
        [
            "California",
            "Alaska",
            "Nevada",
            "Hawaii",
            "Washington",
            "Utah",
            "Montana",
            "Puerto Rico",
            "Indonesia",
            "Chile",
            "Baja California",
            "Oklahoma",
            "Japan",
            "Greece",
            "Papua New Guinea",
            "Philippines",
            "Mexico",
            "Italy",
            "Russia",
            "Idaho",
            "Aleutian Islands",
            "Tonga",
            "Oregon",
            "Wyoming",
            "Turkey",
        ]
    ) & set(df.region.unique())


def load_model() -> cb.CatBoostRegressor:
    path = os.path.join(os.path.dirname(__file__), "../models/earthquake_forecasting_model")
    model = cb.CatBoostRegressor(cat_features=["region"])
    return model.load_model(path)


def reindex(group, delta):
    start_date = group.index.min()
    end_date = pd.Timestamp((datetime.now() + timedelta(days=delta)).date())
    date_range = pd.date_range(start=start_date, end=end_date, freq="d")
    group = group.reindex(date_range)
    group.region = group.region.ffill()
    return group


def preprocess_data(df: pd.DataFrame, region: str) -> pd.DataFrame:
    df = df.copy()

    df["region"] = df.place.str.split(", ", expand=True)[1]
    df.region = df.region.fillna(df.place)
    df.region = df.region.replace({"CA": "California", "B.C.": "Baja California"})

    df.time = pd.to_datetime(df.time)
    df.time = df.time.dt.tz_localize(None)
    df = df.sort_values("time")
    df = df.set_index("time")

    df = df[["depth", "mag", "region"]]

    df = df.groupby("region").resample("d").mean().reset_index()
    df = df.set_index("time")

    df = df.loc[df.region == region]
    start_date = df.index.min()
    end_date = pd.Timestamp(datetime.today().date())
    date_range = pd.date_range(start=start_date, end=end_date, freq="d")
    df = df.reindex(date_range)
    df.region = df.region.ffill()
    df.mag = df.mag.ffill()
    df.depth = df.depth.ffill()

    return df


def create_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()

    start_date = df.index.min()
    end_date = pd.Timestamp((datetime.now() + timedelta(days=horizon)).date())
    date_range = pd.date_range(start=start_date, end=end_date, freq="d")
    df = df.reindex(date_range)
    df.region = df.region.ffill()

    df["day"] = df.index.day
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear

    for i in range(1, _LAGS + 1):
        df[f"mag_lag_{i}"] = df.groupby("region").mag.shift(i)

    for i in range(1, _LAGS + 1):
        df[f"depth_lag_{i}"] = df.groupby("region").depth.shift(i)

    df["mag_ewma"] = df.groupby("region")["mag"].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["depth_ewma"] = df.groupby("region")["depth"].transform(lambda x: x.ewm(span=7, adjust=False).mean())

    return df


def forecast_earthquakes(region: str, days: int) -> list[dict]:
    model = load_model()
    df = get_recent_earthquakes()
    df = preprocess_data(df, region)

    for horizon in range(1, days + 1):
        df = create_features(df, horizon)
        features = (
            [
                "day",
                "dayofweek",
                "dayofyear",
                "mag_ewma",
                "depth_ewma",
            ]
            + [f"mag_lag_{i}" for i in range(1, _LAGS + 1)]
            + [f"depth_lag_{i}" for i in range(1, _LAGS + 1)]
        )
        cat_features = ["region"]
        forecast = model.predict(df[features + cat_features])
        df["mag_forecast"] = forecast[:, 0]
        df["depth_forecast"] = forecast[:, 1]
        df.mag = df.mag.fillna(df.mag_forecast)
        df.depth = df.depth.fillna(df.depth_forecast)

    df = df.reset_index()
    df = df[["index", "mag", "mag_forecast", "depth", "depth_forecast", "region"]]
    df["Latitude"] = df.region.map(_MEDIAN_LATITUDE)
    df["Longitude"] = df.region.map(_MEDIAN_LONGITUDE)
    df = df.rename(
        columns={
            "index": "Date",
            "region": "Region",
            "mag_forecast": "Magnitude Forecast",
            "depth_forecast": "Depth Forecast",
        }
    )
    today = pd.Timestamp.now()
    df = df.loc[df.Date >= today]
    return df.to_dict(orient="records")
