"""Score a local CSV using the estimator produced by run.py; no lending action is taken."""
from pathlib import Path
import argparse,joblib,pandas as pd
from banklab.features import make_features,probability_to_score
from banklab.common import predict,ROOT
p=argparse.ArgumentParser();p.add_argument('input_csv',type=Path);p.add_argument('--output',type=Path,default=ROOT/'artifacts/scored_profiles.csv');args=p.parse_args()
bundle=joblib.load(ROOT/'artifacts/model.joblib')
raw=pd.read_csv(args.input_csv);x=make_features(raw)[bundle['features']];probability=predict(bundle['model'],bundle['calibrator'],x)
result=pd.DataFrame({'input_row':range(len(raw)),'default_probability':probability,'custom_score':probability_to_score(probability)})
args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
print('Wrote',len(result),'research scores to',args.output)
