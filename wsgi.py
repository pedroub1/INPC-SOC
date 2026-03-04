"""
wsgi.py – Despachador WSGI que sirve ambos dashboards SOC desde un solo proceso.

  /      →  soc/dash_app.py     (SOC INPC Mexico)
  /us/   →  soc_us/dash_app.py  (SOC US CPI/PCE)

Inicio local:
    gunicorn wsgi:application --bind 0.0.0.0:8051 --workers 1 --timeout 120

Railway usa este mismo comando con $PORT.
"""
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from soc.dash_app import server as _inpc_server
from soc_us.dash_app import server as _us_server

application = DispatcherMiddleware(_inpc_server, {
    "/us": _us_server,
})
