"""Vercel serverless entrypoint for the Flask forecast UI.

Vercel's @vercel/python runtime serves the module-level WSGI ``app`` variable. The package
lives under ``src/`` (src layout), so we add it to the path before importing the app.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from canola_dt.webapp import app  # noqa: E402  (exposes the WSGI app for Vercel)
