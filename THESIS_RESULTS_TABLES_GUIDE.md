# Chapter 4 results tables in the UI

Start the Streamlit app using `python -m streamlit run dashboard.py`.

Navigate to **Research > Thesis Results Tables**. All five tables from Chapter 4 are visible and exportable as individual CSVs. The original source page can be expanded below each table for direct verification.

- Table 4.1: PDF p. 82 / manuscript p. 81
- Table 4.2: PDF p. 84 / manuscript p. 83
- Table 4.3: PDF p. 86 / manuscript p. 85
- Table 4.4: PDF p. 88 / manuscript p. 87
- Table 4.5: PDF p. 89 / manuscript p. 88

**Do not substitute generated `results/tables` files for the thesis tables.** Some previously generated files describe a different configuration or are mislabeled, so `data/manuscript_tables/` is a separate transcription layer. New experiments are computed independently in Dataset Laboratory / Research Evaluation.

The manuscript's benchmark uses controlled synthetic disk readings combined with Loghub context, and the 85% safety ceiling equals the researcher-defined anomaly labeling threshold. The bootstrapped enhanced-vs-static F1 difference is not statistically distinguishable from zero.

The page also has a collapsed **Additional manuscript tables** section with the original PDF pages for Tables 2.3.1, 2.3.2, 3.1 and 3.2; these tables are not experimental results.
