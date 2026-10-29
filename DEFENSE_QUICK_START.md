# PLM Thesis Defense Dashboard | Quick Start

## Project identity

**Thesis title (cover page):** Enhancement of Threshold-Based Algorithm for Detecting Abnormal Disk Usage Patterns Using System Logs

**Institution:** Pamantasan ng Lungsod ng Maynila (University of the City of Manila), College of Information Systems and Technology Management, Computer Science Department.

**Researchers:** Co, Khurt Luis L.; Bacani, Patrick Jerome D.  
**Adviser:** Prof. Ariel Antwaun Rolando C. Sison  
**Coordinator:** Prof. Raymund M. Dioses  
**Panelists:** Prof. Raymund M. Dioses; Prof. Connie Adriano  
**Thesis cover date:** November 2026

Details are sourced from the supplied thesis PDF (`assets/Thesis_Research_Document.pdf`), title page and approval sheet. The approval sheet has a slightly different title variant.

## Start the real dashboard (Windows)

1. Extract this ZIP and open the `thesis-pipeline-main` folder.
2. Install Python 3.11 (recommended) from https://www.python.org/downloads/ and enable **Add Python to PATH**.
3. Double-click `START_DASHBOARD.bat`. The script creates `.venv`, installs requirements, and starts Streamlit.
4. Visit **http://localhost:8501**. Leave the terminal running during the defense.
5. Press **Ctrl+C** in the terminal when finished.

**Manual alternative (PowerShell in the project folder):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

If Python 3.11 is not installed, replace `py -3.11` with your installed Python command. Internet access is needed on the first installation.

`UI_PREVIEW.html` is an **offline static design preview**, not the running application. The actual Streamlit interface shows the large thesis banner only on **Overview**, and the PLM seal remains visible in the sidebar on all pages. Double-click it to inspect the layout without Python.

## Recommended 5-minute defense demonstration

1. **Overview**: show the PLM logo and presentation-only academic title banner. Then open **Thesis Information** to present school, names, coordinator, adviser, panelists and thesis title; show the cover image.
2. **Research Evaluation**: explain the documented Chapter 4 results and the important differences between suppression-on and priority-only configurations.
3. **Dataset Laboratory**: choose **Example CSV dataset** and use the accompanying example labels. Use the default settings (W=50, Z=5.5, ceiling=85%, context window=30 s).
4. Click **Run detection analysis**, then open **Overview** for the actual computed disk usage chart and alert counts.
5. Open **Alert Intelligence** for ranked alerts and audit download. Revisit **Research Evaluation** for the current run's evaluation.
6. Demonstrate a custom dataset upload (`.csv`, `.log`, `.txt`). To evaluate a new dataset quantitatively, upload independently labeled `.csv` or `.jsonl` ground truth.
7. If requested, turn off **Suppress benign-context alerts** under Detection parameters, rerun, and compare to the suppression-on variant.

## CSV data requirements

Minimum columns:

```csv
timestamp,disk_usage_value,message
2024-03-01 09:00:00,42,normal operation
2024-03-01 09:05:00,46,normal operation
2024-03-01 09:10:00,93,I/O error
```

Ground-truth evaluation labels (optional; must have independently established anomaly status):

```csv
timestamp,is_anomaly
2024-03-01T09:00:00,false
2024-03-01T09:05:00,false
2024-03-01T09:10:00,true
```

**Important**: The example CSV shown above is too short for default baseline warm-up. Use the included `data/examples/demo_disk_usage.csv` with `data/examples/demo_labels.csv` for your first demonstration. Additional log formats supported by the source parser are described in the UI.

## Research accuracy and limitations

- The **documented research metrics** are transcribed from the PDF's Chapter 4, Table 4.4. They do not update when you run a new dataset.
- **Current experiment metrics** are recomputed from the selected run using eligible, timestamp-matched labels; default evaluation excludes warm-up and uses each timestamp once.
- The included Linux benchmark combines an injected, controlled disk usage series with Loghub Linux logs; it is **not proof of production performance**.
- The PDF reports that the priority-only variant matches a static 85% threshold on this benchmark; that 85% value is also used to construct labels. This is a generalization limitation, not evidence of superiority over a tuned static baseline.
- Data processing is **batch upload and evaluation**, not real-time collection from a deployed Linux host. The existing source project also includes separate replay scripts.
- Every experiment is written to `runs/<run-id>/`, leaving archived thesis results unchanged.

## Project modules used

The dashboard imports the existing source code in `src/` for ingestion (Drain3), preprocessing, rolling baseline, Z-score and hybrid safety ceiling, debounce, contextual priority, scoring, ranking and audit history. It does not replace these mechanisms with mock calculations.
