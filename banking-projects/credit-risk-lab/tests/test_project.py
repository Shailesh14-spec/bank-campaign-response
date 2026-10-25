from pathlib import Path
import json,unittest
import numpy as np
import pandas as pd
from banklab.features import make_features,probability_to_score,REQUIRED
from banklab.common import psi
ROOT=Path(__file__).resolve().parents[1]
class FeatureTests(unittest.TestCase):
 def test_score_orientation_and_pdo(self):
  self.assertGreater(probability_to_score([.1])[0],probability_to_score([.2])[0])
  p1=1/51;p2=1/101
  self.assertAlmostEqual(float(probability_to_score([p1])[0]),600)
  self.assertAlmostEqual(float(probability_to_score([p2])[0]-probability_to_score([p1])[0]),20)
 def test_feature_contract(self):
  raw=pd.DataFrame({k:[1.0] for k in REQUIRED});raw['LIMIT_BAL']=100;raw['SEX']=2;raw['AGE']=40
  x=make_features(raw);self.assertNotIn('SEX',x);self.assertNotIn('AGE',x)
  self.assertAlmostEqual(x.utilisation_latest.iloc[0],.01)
  raw.LIMIT_BAL=0
  with self.assertRaises(ValueError):make_features(raw)
 def test_psi_identity(self):
  a=np.arange(100);self.assertAlmostEqual(psi(a,a),0)
class CompletedRunTests(unittest.TestCase):
 def test_split_profile_groups_do_not_overlap(self):
  split=pd.read_csv(ROOT/'artifacts/split_manifest.csv');self.assertEqual(len(split),30000);self.assertTrue(split.source_id.is_unique)
  self.assertEqual(split.groupby('profile_group').split.nunique().max(),1)
  self.assertEqual(set(split.split),{'train','calibration','validation','test'})
 def test_prediction_probabilities_and_summary(self):
  p=pd.read_csv(ROOT/'artifacts/test_predictions.csv');m=json.loads((ROOT/'reports/metrics.json').read_text())
  self.assertTrue(p.pd.between(0,1).all());self.assertEqual(len(p),m['split_rows']['test'])
  self.assertAlmostEqual(float(np.mean((p.actual-p.pd)**2)),m['test']['brier'])
 def test_stress_sensitivity(self):
  s=pd.read_csv(ROOT/'reports/stress_sensitivity.csv')
  for _,group in s.groupby('assumed_lgd'):
   self.assertTrue(group.sort_values('pd_odds_multiplier').one_month_loss_proxy_twd.is_monotonic_increasing)
  for _,group in s.groupby('pd_odds_multiplier'):
   ratios=group.one_month_loss_proxy_twd/group.assumed_lgd;np.testing.assert_allclose(ratios,ratios.iloc[0])
 def test_risk_decile_reconciliation(self):
  d=pd.read_csv(ROOT/'reports/risk_deciles.csv');m=json.loads((ROOT/'reports/metrics.json').read_text());self.assertEqual(d.rows.sum(),m['split_rows']['test'])
if __name__=='__main__':unittest.main()
