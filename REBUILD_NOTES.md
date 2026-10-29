# PLM Defense Dashboard — Rebuilt UI

## Changes

- Replaced the previous visual theme with a simpler academic navy interface.
- Integrated the user-supplied official PLM seal (`assets/plm-logo.png`) in sidebar branding and the Overview banner.
- The academic overview banner is rendered only when the Overview navigation page is active.
- Added organized navigation: Overview, Thesis Information, Dataset Laboratory, Monitoring Overview, Alert Intelligence, Methodology, Research Evaluation, Settings.
- Preserved CSV/LOG/TXT upload testing, optional independently labeled data, isolated experiment runs, detection parameters, alerts, downloads and comparative evaluation.
- No placeholder statistics are presented as computed experiment results. Empty cards show em dashes until a real dataset is processed.
- The HTML file `UI_PREVIEW.html` is an offline visual preview only; it uses example raw CSV values and documented thesis F1 scores, not current alerts.

## Quick commands (VS Code PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open http://localhost:8501. Alternatively double-click `START_DASHBOARD.bat`.

## Test limits

The Python code compiles and the original detection modules passed a smoke test with the example CSV in both correlation policies using a temporary stand-in for the unavailable Drain3 library. The browser-based Streamlit runtime must be verified in your local Python environment after installing requirements.
