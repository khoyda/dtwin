"""Smoke test for the end-to-end twin on synthetic weather."""

from canola_dt.config import load_config
from canola_dt.data.ingest import synthetic_weather
from canola_dt.simulation.twin import CanolaDigitalTwin


def test_twin_runs_without_model():
    cfg = load_config()
    twin = CanolaDigitalTwin(cfg)
    weather = synthetic_weather(year=2023, n_days=130, seed=1)
    result = twin.run(weather)

    assert len(result.state_trajectory) == 130
    assert result.predicted_yield_kg_ha is None  # no model attached
    assert result.features["total_gdd"] > 0
    # Cumulative GDD should be non-decreasing across the season.
    cum = result.state_trajectory["cum_gdd"]
    assert list(cum) == sorted(cum)
