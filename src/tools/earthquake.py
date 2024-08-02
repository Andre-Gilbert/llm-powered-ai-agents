"""Earthquake tools."""

from datetime import datetime, timedelta
from typing import Any

import requests
from pydantic import BaseModel, Field


class USGeopoliticalSurveyEarthquakeAPI(BaseModel):
    """Class that implements the API interface."""

    start_time: str = Field(
        None,
        description=(
            "Limit to events on or after the specified start time. NOTE: All times use ISO8601 Date/Time format."
            + " Unless a timezone is specified, UTC is assumed."
        ),
    )
    end_time: str = Field(
        None,
        description=(
            "Limit to events on or before the specified end time. NOTE: All times use ISO8601 Date/Time format."
            + " Unless a timezone is specified, UTC is assumed."
        ),
    )
    limit: int = Field(
        20000,
        description=(
            "Limit the results to the specified number of events. NOTE: The service limits queries to 20000,"
            + " and any that exceed this limit will generate a HTTP response code 400 Bad Request."
        ),
    )
    min_depth: int = Field(
        -100,
        description="Limit to events with depth more than the specified minimum.",
    )
    max_depth: int = Field(
        1000,
        description="Limit to events with depth less than the specified maximum.",
    )
    min_magnitude: int = Field(
        None,
        description="Limit to events with a magnitude larger than the specified minimum.",
    )
    max_magnitude: int = Field(
        None,
        description="Limit to events with a magnitude smaller than the specified maximum.",
    )
    alert_level: str = Field(
        None,
        description=(
            "Limit to events with a specific PAGER alert level."
            + " The allowed values are: alert_level=green Limit to events with PAGER"
            + ' alert level "green". alert_level=yellow Limit to events with PAGER alert level "yellow".'
            + ' alert_level=orange Limit to events with PAGER alert level "orange".'
            + ' alert_level=red Limit to events with PAGER alert level "red".'
        ),
    )


def query_earthquakes(
    start_time: datetime = (datetime.now() - timedelta(days=30)).date(),
    end_time: datetime = datetime.now().date(),
    limit: int = 20000,
    min_depth: int = -100,
    max_depth: int = 1000,
    min_magnitude: int | None = None,
    max_magnitude: int | None = None,
    alert_level: str | None = None,
) -> Any:
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "limit": limit,
        "mindepth": min_depth,
        "maxdepth": max_depth,
        "minmagnitude": min_magnitude,
        "maxmagnitude": max_magnitude,
        "alertlevel": alert_level,
        "eventtype": "earthquake",
    }
    response = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params,
        timeout=None,
    )
    return response.json()


def count_earthquakes(
    start_time: datetime = (datetime.now() - timedelta(days=30)).date(),
    end_time: datetime = datetime.now().date(),
    limit: int = 20000,
    min_depth: int = -100,
    max_depth: int = 1000,
    min_magnitude: int | None = None,
    max_magnitude: int | None = None,
    alert_level: str | None = None,
) -> int:
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "limit": limit,
        "mindepth": min_depth,
        "maxdepth": max_depth,
        "minmagnitude": min_magnitude,
        "maxmagnitude": max_magnitude,
        "alertlevel": alert_level,
        "eventtype": "earthquake",
    }
    return requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/count",
        params=params,
        timeout=None,
    ).json()
