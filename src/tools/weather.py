"""Weather lookup via Open-Meteo API (no API key)."""

from __future__ import annotations

import time
from urllib.parse import urlencode

import requests

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

WEATHER_SCHEMA = {
    "properties": {
        "location": {"type": "string", "description": "City name or 'city, country', e.g. London, Paris, New York"},
    },
    "required": ["location"],
}


def _geocode(location: str) -> tuple[float, float] | None:
    """Resolve location to lat,lon using Open-Meteo geocoding."""
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        return None
    return float(results[0]["latitude"]), float(results[0]["longitude"])


@tool_registry.register(
    name="weather",
    description="Get current weather for a city or location. Uses Open-Meteo (no API key).",
    category="information_retrieval",
    parameters_schema=WEATHER_SCHEMA,
)
async def weather(location: str) -> ToolResult:
    start = time.perf_counter()
    try:
        coords = _geocode(location)
        if not coords:
            return ToolResult(success=False, data=None, error=f"Location not found: {location}", execution_time_ms=(time.perf_counter() - start) * 1000)
        lat, lon = coords
        params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m", "timezone": "auto"}
        r = requests.get("https://api.open-meteo.com/v1/forecast?" + urlencode(params), timeout=10)
        r.raise_for_status()
        data = r.json()
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        summary = f"Current weather in {location}: {temp}°C, humidity {humidity}%, wind {wind} km/h."
        return ToolResult(
            success=True,
            data={
                "location": location,
                "temperature_c": temp,
                "humidity_percent": humidity,
                "wind_speed_kmh": wind,
                "weather_code": current.get("weather_code"),
                "summary": summary,
            },
            metadata={"summary": summary},
            execution_time_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        logger.exception("weather_failed", location=location, error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)
