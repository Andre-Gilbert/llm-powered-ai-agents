"""Forecasting tool."""

import os
from urllib import parse

import pandas as pd
from catboost import CatBoostRegressor

_PARAMS = {"format": "csv", "eventtype": "earthquake", "limit": 20000}


def get_earthquakes_data(base_url: str, start_time: str, end_time: str) -> pd.DataFrame:
    _PARAMS.update({"starttime": start_time, "endtime": end_time})
    url = build_url(base_url, _PARAMS)
    return pd.read_csv(url)


def build_url(base_url: str, params: dict[str, str]) -> str:
    return base_url + parse.urlencode(params)


_MODEL_FILE = os.path.join(os.path.dirname(__file__), "earthquake_forecasting_model")


class MLModel:
    """ML model that predicts the magnitude of earthquakes."""

    _FEATURES = [
        "dayofyear",
        "hour",
        "dayofweek",
        "month",
        "season",
        "year",
        "mag_5eq_lag",
        "mag_10eq_lag",
        "mag_15eq_lag",
        "mag_5eq_avg",
        "mag_10eq_avg",
        "mag_15eq_avg",
        "mag_5eq_min",
        "mag_10eq_min",
        "mag_15eq_min",
        "mag_5eq_max",
        "mag_10eq_max",
        "mag_15eq_max",
        "mag_5eq_std",
        "mag_10eq_std",
        "mag_15eq_std",
        "depth_5eq_lag",
        "depth_10eq_lag",
        "depth_15eq_lag",
        "depth_5eq_avg",
        "depth_10eq_avg",
        "depth_15eq_avg",
        "depth_5eq_min",
        "depth_10eq_min",
        "depth_15eq_min",
        "depth_5eq_max",
        "depth_10eq_max",
        "depth_15eq_max",
        "depth_5eq_std",
        "depth_10eq_std",
        "depth_15eq_std",
        "latitude",
        "longitude",
        "location",
    ]

    def __init__(self):
        self._model = CatBoostRegressor(cat_features="location")
        self._model.load_model(_MODEL_FILE)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df.mag.notna()]
        df.time = pd.to_datetime(df.time)
        df["location"] = df.place
        df = df[::-1]
        df = self._preprocess_data(df)
        prediction = self._model.predict(df[self._FEATURES]).round(6)
        df_pred = pd.DataFrame(
            {
                "time": df.time,
                "prediction": prediction,
                "latitude": df.latitude,
                "longitude": df.longitude,
                "mag": df.mag,
                "id": df.id,
                "place": df.place,
                "location": df.location,
            }
        )
        return df_pred.sort_values(by="time")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = []
        for location in df.location.unique():
            temp = df.loc[df.location == location]
            temp = self._create_features(temp)
            temp = self._add_lags(temp)
            temp = self._add_rolling_windows(temp)
            data.append(temp)
        df = pd.concat(data)
        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df.time.dt.hour
        df["dayofweek"] = df.time.dt.dayofweek
        df["month"] = df.time.dt.month
        df["year"] = df.time.dt.year
        df["dayofyear"] = df.time.dt.dayofyear
        df["dayofmonth"] = df.time.dt.day
        df["weekofyear"] = df.time.dt.isocalendar().week
        df["quarter"] = df.time.dt.quarter
        df["season"] = df.month % 12 // 3 + 1
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        df["mag_5eq_lag"] = df.mag.shift(5)
        df["mag_10eq_lag"] = df.mag.shift(10)
        df["mag_15eq_lag"] = df.mag.shift(15)
        df["depth_5eq_lag"] = df.depth.shift(5)
        df["depth_10eq_lag"] = df.depth.shift(10)
        df["depth_15eq_lag"] = df.depth.shift(15)
        return df

    def _add_rolling_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        df["mag_5eq_avg"] = df.mag.rolling(window=5, center=False).mean()
        df["mag_10eq_avg"] = df.mag.rolling(window=10, center=False).mean()
        df["mag_15eq_avg"] = df.mag.rolling(window=15, center=False).mean()
        df["mag_5eq_min"] = df.mag.rolling(window=5, center=False).min()
        df["mag_10eq_min"] = df.mag.rolling(window=10, center=False).min()
        df["mag_15eq_min"] = df.mag.rolling(window=15, center=False).min()
        df["mag_5eq_max"] = df.mag.rolling(window=5, center=False).max()
        df["mag_10eq_max"] = df.mag.rolling(window=10, center=False).max()
        df["mag_15eq_max"] = df.mag.rolling(window=15, center=False).max()
        df["mag_5eq_std"] = df.mag.rolling(window=5, center=False).std()
        df["mag_10eq_std"] = df.mag.rolling(window=10, center=False).std()
        df["mag_15eq_std"] = df.mag.rolling(window=15, center=False).std()
        df["depth_5eq_avg"] = df.depth.rolling(window=5, center=False).mean()
        df["depth_10eq_avg"] = df.depth.rolling(window=10, center=False).mean()
        df["depth_15eq_avg"] = df.depth.rolling(window=15, center=False).mean()
        df["depth_5eq_min"] = df.depth.rolling(window=5, center=False).min()
        df["depth_10eq_min"] = df.depth.rolling(window=10, center=False).min()
        df["depth_15eq_min"] = df.depth.rolling(window=15, center=False).min()
        df["depth_5eq_max"] = df.depth.rolling(window=5, center=False).max()
        df["depth_10eq_max"] = df.depth.rolling(window=10, center=False).max()
        df["depth_15eq_max"] = df.depth.rolling(window=15, center=False).max()
        df["depth_5eq_std"] = df.depth.rolling(window=5, center=False).std()
        df["depth_10eq_std"] = df.depth.rolling(window=10, center=False).std()
        df["depth_15eq_std"] = df.depth.rolling(window=15, center=False).std()
        return df


ml_model = MLModel()
