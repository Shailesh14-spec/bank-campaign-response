from pathlib import Path
from urllib.request import urlopen
import zipfile,io,hashlib
ROOT=Path(__file__).resolve().parent
URL='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
EXPECTED='30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933'
with urlopen(URL,timeout=90) as response:payload=response.read()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:data=archive.read('default of credit card clients.xls')
if hashlib.sha256(data).hexdigest()!=EXPECTED:raise ValueError('Source checksum changed; review upstream before updating')
(ROOT/'data').mkdir(exist_ok=True);(ROOT/'data/credit_default.xls').write_bytes(data)
print('Verified UCI credit-default workbook. Attribution: docs/DATA_SOURCE.md')
