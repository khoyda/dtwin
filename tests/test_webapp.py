"""Tests for the Flask scenario UI (offline via synthetic weather)."""

import pytest

pytest.importorskip("flask")
from canola_dt.webapp import app  # noqa: E402


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def test_form_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Prairie Crop Digital Twin" in r.data


def test_wheat_forecast_renders_result(client):
    r = client.post("/", data={"crop": "wheat", "province": "Saskatchewan",
                               "weather": "synthetic", "analog_year": "2022", "n": "110"})
    assert r.status_code == 200
    assert b"t/ha" in r.data and b"bu/ac" in r.data
    assert b"limited by" in r.data


def test_barley_malt_grade_shown(client):
    r = client.post("/", data={"crop": "barley", "weather": "synthetic",
                               "variety": "malt_2row", "n": "160"})
    assert r.status_code == 200
    assert b"malt" in r.data.lower()
