"""Financial-history features. Demographic fields and row ID are not model inputs."""
import numpy as np
import pandas as pd
PAY=['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
BILL=[f'BILL_AMT{i}' for i in range(1,7)]
PAID=[f'PAY_AMT{i}' for i in range(1,7)]
REQUIRED=['LIMIT_BAL']+PAY+BILL+PAID
EXCLUDED=['ID','SEX','AGE','EDUCATION','MARRIAGE','default payment next month']
def make_features(raw):
 missing=set(REQUIRED)-set(raw)
 if missing:raise ValueError('Missing columns: '+str(sorted(missing)))
 if raw[REQUIRED].isna().any().any():raise ValueError('Missing financial inputs')
 if (raw.LIMIT_BAL<=0).any():raise ValueError('LIMIT_BAL must be positive')
 x=raw[REQUIRED].astype(float).copy()
 # Preserve delinquency sentinel values; no undocumented recoding.
 x['utilisation_latest']=raw.BILL_AMT1/raw.LIMIT_BAL
 x['utilisation_mean6']=raw[BILL].mean(axis=1)/raw.LIMIT_BAL
 x['positive_delay_months']=(raw[PAY]>0).sum(axis=1)
 x['maximum_delay']=raw[PAY].max(axis=1)
 x['bill_change']=raw.BILL_AMT1-raw.BILL_AMT6
 x['payment_to_bill']=raw.PAY_AMT1/np.maximum(raw.BILL_AMT1.abs(),1)
 return x
def probability_to_score(p,base_score=600,base_good_bad_odds=50,pdo=20):
 p=np.clip(np.asarray(p),1e-6,1-1e-6)
 return base_score+pdo/np.log(2)*np.log(((1-p)/p)/base_good_bad_odds)
