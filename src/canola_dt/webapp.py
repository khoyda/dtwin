"""Simple one-page web UI for canola/wheat/barley scenario forecasting (Flask).

Run via ``python scripts/forecast_ui.py`` and open http://127.0.0.1:5000. A form over
:func:`canola_dt.scenario.run_scenario`: pick a crop, weather basis and management plan;
get yield, protein, limiting factor, a fertility recommendation and planning alerts.
"""

from __future__ import annotations

from flask import Flask, render_template_string, request

from canola_dt.config import load_config
from canola_dt.scenario import Scenario, run_scenario

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
    <select name="crop">{% for c in ['canola','wheat','barley'] %}
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
    <button type="submit">Forecast →</button>
    <p class="muted">inseason = current-year weather to date + analog fill. First run may take a few seconds (fetches ECCC).</p>
  </form>

  <div>
  {% if error %}<div class="card err">{{error}}</div>{% endif %}
  {% if result %}
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


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        try:
            sc = Scenario(
                crop=form.get("crop", "wheat"),
                name=(form.get("name") or f"{form.get('crop', 'wheat')} forecast"),
                province=form.get("province", "Saskatchewan"),
                weather=form.get("weather", "inseason"),
                analog_year=int(form.get("analog_year") or 2022),
                preceding_crop=form.get("preceding", "") or "",
                variety=form.get("variety", "") or "",
                plants_per_m2=_num(form.get("plants")),
                n=_num(form.get("n")), s=_num(form.get("s")),
            )
            result = run_scenario(sc, _cfg())
        except Exception as e:  # surface any failure to the page rather than 500
            error = f"{type(e).__name__}: {e}"
    return render_template_string(TEMPLATE, result=result, error=error, form=form)


def main(host: str = "127.0.0.1", port: int = 5000) -> None:
    print(f"Prairie Crop Digital Twin UI → http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
