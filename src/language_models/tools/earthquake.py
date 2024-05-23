"""Earthquake tools."""

from datetime import datetime, timedelta

import requests
from pydantic import BaseModel, Field

from language_models.tools.tool import Tool


class USGSEarthquakeAPI(BaseModel):
    start_time: str = Field(
        None,
        description=(
            "Limit to events on or after the specified start time. NOTE: All times use ISO8601 Date/Time format."
            + " Unless a timezone is specified, UTC is assumed. Default: NOW - 30 days."
        ),
    )
    end_time: str = Field(
        None,
        description=(
            "Limit to events on or before the specified end time. NOTE: All times use ISO8601 Date/Time format."
            + " Unless a timezone is specified, UTC is assumed. Default: present time."
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
    eventtype: str = Field(
        None,
        description="Limit to events of a specific type. NOTE: “earthquake” will filter non-earthquake events.",
    )


def query(
    start_time=None,
    end_time=None,
    limit=20000,
    min_depth=-100,
    max_depth=1000,
    min_magnitude=None,
    max_magnitude=None,
    alert_level=None,
    eventtype=None,
):
    if start_time is None:
        start_time = (datetime.now() - timedelta(days=30)).date()
    if end_time is None:
        end_time = datetime.now().date()
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
        "eventtype": eventtype,
    }
    response = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params,
        timeout=None,
    )
    return response.json()


def count(
    start_time=None,
    end_time=None,
    limit=20000,
    min_depth=-100,
    max_depth=1000,
    min_magnitude=None,
    max_magnitude=None,
    alert_level=None,
    eventtype=None,
):
    if start_time is None:
        start_time = (datetime.now() - timedelta(days=30)).date()
    if end_time is None:
        end_time = datetime.now().date()
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
        "eventtype": eventtype,
    }
    response = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/count",
        params=params,
        timeout=None,
    )
    return response.json()


earthquake_tools = [
    Tool(
        func=query,
        name="Query Earthquakes",
        description="Use this tool to search recent earthquakes.",
        args_schema=USGSEarthquakeAPI,
    ),
    Tool(
        func=count,
        name="Count Earthquakes",
        description="Use this tool to count and aggregate recent earthquakes.",
        args_schema=USGSEarthquakeAPI,
    ),
]
