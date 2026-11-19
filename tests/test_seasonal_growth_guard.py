import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from seasonal_growth_guard import SRGGConfig, apply_srgg


class SRGGTests(unittest.TestCase):
    start = datetime(2024, 1, 1)

    def make(self, n=1200, usage=None):
        if usage is None:
            usage = [45.] * n
        return [{'timestamp': self.start + timedelta(minutes=5*i),
                 'disk_usage_value': float(v), 'classification':'NORMAL',
                 'zscore':0., 'severity_score':0., 'triggered_by':None}
                for i,v in enumerate(usage)]

    def test_stable_data_no_guard(self):
        a = apply_srgg(self.make())
        self.assertFalse(any(r['srgg_alert'] for r in a))

    def test_persistent_growth_can_flag(self):
        vals = [45.] * 950 + [45. + (i-949)*0.35 for i in range(950,1200)]
        rows = apply_srgg(self.make(1200,vals))
        self.assertTrue(any(r['srgg_new_detection'] for r in rows))

    def test_single_spike_not_persistent(self):
        vals = [45.] * 1200; vals[1050] = 90.
        rows = apply_srgg(self.make(1200,vals))
        self.assertFalse(any(r['srgg_alert'] for r in rows))

    def test_old_critical_alert_is_preserved(self):
        a = self.make(); a[30]['classification'] = 'ABNORMAL';a[30]['triggered_by']='ceiling'
        out = apply_srgg(a)
        self.assertEqual(out[30]['classification'],'ABNORMAL')
        self.assertEqual(out[30]['triggered_by'],'ceiling')

    def test_no_early_lookahead(self):
        a = self.make(); a[-1]['disk_usage_value'] = 99
        out = apply_srgg(a)
        self.assertFalse(any(r['srgg_alert'] for r in out[:864]))

    def test_gap_resets_streak(self):
        a = self.make()
        for i in range(950,len(a)):
            a[i]['disk_usage_value'] = 45 + (i-949)*0.35
        a[1100]['timestamp'] += timedelta(minutes=15)
        out = apply_srgg(a)
        self.assertEqual(out[1100]['srgg_streak'],0)
        self.assertEqual(out[1100]['srgg_eligible'],False)

    def test_one_day_ablation(self):
        a = apply_srgg(self.make(), SRGGConfig(reference_days=(1,),persistence_readings=1))
        self.assertFalse(any(r['srgg_alert'] for r in a))

if __name__ == '__main__':unittest.main()
