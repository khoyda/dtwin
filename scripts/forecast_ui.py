"""Launch the simple web UI for scenario forecasting.

    python scripts/forecast_ui.py        # serves http://127.0.0.1:5000

Requires Flask (pip install -e ".[ui]"  or  pip install flask).
"""

from canola_dt.webapp import main

if __name__ == "__main__":
    main()
