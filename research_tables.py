"""Published Chapter Four tables transcribed verbatim from the uploaded thesis PDF.

These are static, documented results -- NEVER results from user-uploaded datasets.
The CSV files intentionally preserve display precision and strings from the paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'manuscript_tables'
PAGE_IMAGES = ROOT / 'assets' / 'research_table_pages'

@dataclass(frozen=True)
class PublishedTable:
    code: str
    title: str
    manuscript_page: int
    pdf_page: int
    note: str
    
    @property
    def csv_path(self) -> Path:
        return DATA_DIR / f'table_{self.code.replace(".", "_")}.csv'

    @property
    def page_image(self) -> Path:
        return PAGE_IMAGES / f'table_{self.code.replace(".", "_")}_source.png'


TABLES = (
    PublishedTable('4.1','Z-Score Threshold Sensitivity Sweep (W = 50, no ceiling)',81,82,
        'The Z-score-only detector improves precision as θ_z rises, but its best F1 (0.2069 at θ_z = 4.0) remains below the static 80% reference (0.7156).'),
    PublishedTable('4.2','Window-Size Sweep (θ_z = 2.0, no ceiling)',83,84,
        'At the default θ_z = 2.0, increasing W does not consistently raise recall. All listed F1 scores remain at or below 0.0036.'),
    PublishedTable('4.3','Joint θ_z × θ_ceiling Sensitivity Sweep (selected results; 63 combinations run)',85,86,
        'At θ_z = 5.5 and ceiling = 85%, the suppression-enabled detector has TP = 27, FP = 0, FN = 18, and F1 = 0.7500. The table shows 11 selected parameter combinations plus the static 80% reference.'),
    PublishedTable('4.4','Final Comparative Performance of Static Threshold vs. Enhanced Algorithm',87,88,
        'Suppression removes all 25 false positives of the static 80% comparator but reduces recall to 0.6000. Priority-only hybrid and static 85% both attain F1 = 0.9286 on this labeled benchmark.'),
    PublishedTable('4.5','Bootstrap 95% Confidence Intervals (5,000 resamples)',88,89,
        'The paired F1 difference (Enhanced − Static 80%) has a 95% confidence interval of [−0.095, 0.164]; the thesis does not establish a statistically distinguishable F1 advantage. The hybrid was not bootstrapped.'),
)

def load_published_table(table: PublishedTable) -> pd.DataFrame:
    """Read source transcription without parsing display values into floats."""
    return pd.read_csv(table.csv_path, dtype=str, keep_default_na=False)
