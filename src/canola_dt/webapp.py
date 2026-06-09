"""Simple one-page web UI for canola/wheat/barley scenario forecasting (Flask).

Run via ``python scripts/forecast_ui.py`` and open http://127.0.0.1:5000. A form over
:func:`canola_dt.scenario.run_scenario`: pick a crop, weather basis and management plan;
get yield, protein, limiting factor, a fertility recommendation and planning alerts.
"""

from __future__ import annotations

import base64
import io
import os

# Serverless hosts (e.g. Vercel) have a read-only HOME; point matplotlib's cache at /tmp.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

try:  # matplotlib is only needed for the N-sweep chart; degrade gracefully without it
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")  # headless server-side rendering
    import matplotlib.pyplot as plt  # noqa: E402
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from flask import Flask, render_template_string, request  # noqa: E402

from canola_dt.config import load_config  # noqa: E402
from canola_dt.scenario import Scenario, run_scenario  # noqa: E402

app = Flask(__name__)
_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _num(value):
    value = (value or "").strip()
    return float(value) if value else None


def _yield_n_chart(sweep: list[dict], crop: str) -> str | None:
    """Render yield (and protein) vs N as a base64 PNG for inline embedding."""
    if not _HAS_MPL:
        return None
    ns = [r["n_applied"] for r in sweep]
    ys = [r["yield_t_ha"] for r in sweep]
    ps = [r["protein_pct"] for r in sweep]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(ns, ys, marker="o", color="#2e7d32", label="yield")
    ax.set_xlabel("N applied (kg/ha)")
    ax.set_ylabel("yield (t/ha)", color="#2e7d32")
    ax.grid(alpha=0.25)
    if any(p is not None for p in ps):
        ax2 = ax.twinx()
        ax2.plot(ns, ps, marker="s", color="#b06a00", label="protein")
        ax2.set_ylabel("protein (%)", color="#b06a00")
        if crop == "barley":  # malt acceptance ceiling
            ax2.axhline(12.5, ls="--", lw=1, color="#b06a00", alpha=0.6)
            ax2.text(ns[0], 12.6, "malt max 12.5%", color="#b06a00", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Prairie Crop Digital Twin</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--g:#2e7d32;--bg:#f4f6f4;--bd:#d9ddd9;}
  body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:var(--bg);color:#1b211b}
  header{background:var(--g);color:#fff;padding:16px 24px}
  header h1{margin:0;font-size:20px} header p{margin:4px 0 0;opacity:.9;font-size:13px}
  main{max-width:960px;margin:20px auto;padding:0 16px;display:grid;grid-template-columns:320px 1fr;gap:20px}
  @media(max-width:760px){main{grid-template-columns:1fr}}
  .card{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:16px}
  label{display:block;font-size:12px;font-weight:600;margin:10px 0 3px;color:#3a463a}
  input,select{width:100%;padding:7px 8px;border:1px solid var(--bd);border-radius:6px;font-size:14px;box-sizing:border-box}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  button{margin-top:16px;width:100%;background:var(--g);color:#fff;border:0;border-radius:6px;padding:11px;font-size:15px;font-weight:600;cursor:pointer}
  button:hover{background:#1f5c23}
  .big{font-size:30px;font-weight:700;color:var(--g)} .sub{color:#5a665a;font-size:13px}
  table{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}
  td{padding:5px 4px;border-bottom:1px solid #eee} td:first-child{color:#5a665a}
  .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600}
  .ok{background:#e3f3e4;color:#256b29} .bad{background:#fbe3e3;color:#a32626} .warn{background:#fff3df;color:#9a6a12}
  .alert{font-size:13px;padding:6px 8px;border-radius:6px;margin-top:6px;background:#fafafa;border-left:4px solid #bbb}
  .err{background:#fbe3e3;color:#a32626;padding:10px;border-radius:6px}
  .muted{color:#7a847a;font-size:12px;margin-top:14px}
</style></head><body>
<header><h1>🌾 Prairie Crop Digital Twin</h1>
<p>Scenario forecasting — canola · spring wheat · spring barley (calibrated, 2021 drought excluded)</p></header>
<main>
  <form class="card" method="post">
    <label>Crop</label>
    <select name="crop">{% for c in ['canola','wheat','barley','pea'] %}
      <option value="{{c}}" {{'selected' if form.get('crop')==c else ''}}>{{c}}</option>{% endfor %}</select>
    <label>Province</label>
    <select name="province">{% for p in ['Saskatchewan','Alberta','Manitoba'] %}
      <option {{'selected' if form.get('province')==p else ''}}>{{p}}</option>{% endfor %}</select>
    <div class="row"><div>
      <label>Weather basis</label>
      <select name="weather">{% for w in ['inseason','analog','synthetic'] %}
        <option value="{{w}}" {{'selected' if form.get('weather')==w else ''}}>{{w}}</option>{% endfor %}</select>
    </div><div>
      <label>Analog year</label>
      <input name="analog_year" type="number" value="{{form.get('analog_year','2022')}}">
    </div></div>
    <div class="row"><div>
      <label>N applied (kg/ha)</label><input name="n" type="number" step="any" value="{{form.get('n','')}}" placeholder="default">
    </div><div>
      <label>S applied (kg/ha)</label><input name="s" type="number" step="any" value="{{form.get('s','')}}" placeholder="default">
    </div></div>
    <label>Plant density / population (per m²)</label>
    <input name="plants" type="number" step="any" value="{{form.get('plants','')}}" placeholder="default">
    <label>Preceding crop <span class="sub">(canola: peas/wheat/canola · wheat/barley: canola/pulse/cereal/wheat/barley)</span></label>
    <input name="preceding" value="{{form.get('preceding','')}}" placeholder="default">
    <label>Variety / class <span class="sub">(canola: hybrid · wheat: CWRS · barley: malt_2row/feed)</span></label>
    <input name="variety" value="{{form.get('variety','')}}" placeholder="default">
    <label>Compare N rates <span class="sub">(comma-separated → chart, e.g. 60,90,120,150)</span></label>
    <input name="n_sweep" value="{{form.get('n_sweep','')}}" placeholder="leave blank for single forecast">
    <button type="submit">Forecast →</button>
    <p class="muted">inseason = current-year weather to date + analog fill. First run may take a few seconds (fetches ECCC).</p>
  </form>

  <div>
  {% if error %}<div class="card err">{{error}}</div>{% endif %}
  {% if sweep %}
    <div class="card">
      <div class="sub">{{sweep_label}}</div>
      {% if chart %}<img src="data:image/png;base64,{{chart}}" alt="yield vs N" style="width:100%;margin-top:8px">{% endif %}
      <table>
        <tr><td><b>N kg/ha</b></td><td><b>yield t/ha</b></td><td><b>bu/ac</b></td>
            <td><b>protein</b></td><td><b>limited by</b></td>{% if sweep[0].malt_grade_ok is not none %}<td><b>malt</b></td>{% endif %}</tr>
        {% for r in sweep %}<tr>
          <td>{{r.n_applied}}</td><td>{{r.yield_t_ha}}</td><td>{{r.yield_bu_ac}}</td>
          <td>{{r.protein_pct if r.protein_pct is not none else '—'}}</td>
          <td>{{r.limiting_factor}}</td>
          {% if r.malt_grade_ok is not none %}<td><span class="pill {{'ok' if r.malt_grade_ok else 'bad'}}">{{'OK' if r.malt_grade_ok else 'FAIL'}}</span></td>{% endif %}
        </tr>{% endfor %}
      </table>
    </div>
  {% elif result %}
    <div class="card">
      <div class="sub">{{result.name}} · {{result.crop}} · {{result.weather}}</div>
      <div class="big">{{result.yield_t_ha}} t/ha <span class="sub">({{result.yield_bu_ac}} bu/ac)</span></div>
      <span class="pill {{'bad' if result.limiting_factor!='water/weather' else 'ok'}}">limited by: {{result.limiting_factor}}</span>
      {% if result.malt_grade_ok is not none %}
        <span class="pill {{'ok' if result.malt_grade_ok else 'bad'}}">malt: {{'OK' if result.malt_grade_ok else 'FAIL → feed'}}</span>
      {% endif %}
      <table>
        <tr><td>Biophysical (water/weather)</td><td>{{result.biophysical_t_ha}} t/ha</td></tr>
        {% if result.protein_pct is not none %}<tr><td>Estimated protein</td><td>{{result.protein_pct}} %</td></tr>{% endif %}
        <tr><td>Days to {{'flowering' if result.crop=='canola' else 'heading/anthesis'}}</td><td>{{result.days_to_flower}}</td></tr>
        <tr><td>Days to maturity</td><td>{{result.days_to_maturity}} {{'' if result.reached_maturity else '(not reached)'}}</td></tr>
        <tr><td>Fertilizer rec (N/P₂O₅/K₂O/S)</td><td>{{result.fertilizer_kg_ha['N']}} / {{result.fertilizer_kg_ha['P2O5']}} / {{result.fertilizer_kg_ha['K2O']}} / {{result.fertilizer_kg_ha['S']}} kg/ha</td></tr>
        <tr><td>Limiting nutrient</td><td>{{result.limiting_nutrient}}</td></tr>
      </table>
      {% if result.alerts %}<h4 style="margin:14px 0 4px">Planning alerts</h4>
        {% for a in result.alerts %}<div class="alert">{{a}}</div>{% endfor %}{% endif %}
    </div>
  {% else %}{% if not error %}
    <div class="card sub">Set a scenario on the left and press <b>Forecast</b>. Try the same crop at two N
    rates, or canola with low S, to see yield vs protein vs the limiting nutrient.</div>
  {% endif %}{% endif %}
  </div>
</main></body></html>"""


def _scenario(form, n_override=None) -> Scenario:
    crop = form.get("crop", "wheat")
    return Scenario(
        crop=crop, name=(form.get("name") or f"{crop} forecast"),
        province=form.get("province", "Saskatchewan"),
        weather=form.get("weather", "inseason"),
        analog_year=int(form.get("analog_year") or 2022),
        preceding_crop=form.get("preceding", "") or "",
        variety=form.get("variety", "") or "",
        plants_per_m2=_num(form.get("plants")),
        n=(n_override if n_override is not None else _num(form.get("n"))),
        s=_num(form.get("s")),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = chart = sweep_label = None
    sweep: list[dict] = []
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        crop = form.get("crop", "wheat")
        n_sweep = (form.get("n_sweep") or "").strip()
        try:
            if n_sweep:  # compare mode: vary N, chart yield + protein
                n_values = sorted({float(x) for x in n_sweep.replace(" ", "").split(",") if x})
                for nv in n_values:
                    r = run_scenario(_scenario(form, nv), _cfg())
                    r["n_applied"] = nv
                    sweep.append(r)
                chart = _yield_n_chart(sweep, crop)
                sweep_label = f"{crop} · N sweep · {sweep[0]['weather']}"
            else:
                result = run_scenario(_scenario(form), _cfg())
        except Exception as e:  # surface any failure to the page rather than 500
            error = f"{type(e).__name__}: {e}"
    return render_template_string(TEMPLATE, result=result, sweep=sweep, chart=chart,
                                  sweep_label=sweep_label, error=error, form=form)


def main(host: str = "127.0.0.1", port: int = 5000) -> None:
    print(f"Prairie Crop Digital Twin UI → http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
