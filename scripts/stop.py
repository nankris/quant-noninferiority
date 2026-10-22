"""Stop only this experiment after complete, locally collected measurements."""
from pathlib import Path
import json,subprocess,time
r=Path('results')
if (r/'resource_end.json').exists():
    print('Stop already recorded');raise SystemExit()
for method in ['bf16','gptq','awq','hqq']:
    meta=json.loads((r/f'{method}_done.json').read_text())
    rows=json.loads((r/f'{method}_predictions.json').read_text())
    assert sorted(x['item_id'] for x in rows)==list(range(15214))
    assert len(meta['generation_seconds'])==3 and meta['tokens_per_second']>0
begin=time.time()
p=subprocess.run(['colab','stop','-s','quant-noninferiority'],capture_output=True,text=True)
(r/'stop.txt').write_text(p.stdout+p.stderr)
p.check_returncode()
end=time.time()
budget=json.loads((r/'budget.json').read_text())
(r/'resource_end.json').write_text(json.dumps({'session':'quant-noninferiority','stop_request_unix':begin,'stop_unix':end,'allocation_unix':budget['allocation_unix']},indent=2))
print(p.stdout+p.stderr)
u=subprocess.run(['colab','usage'],capture_output=True,text=True,timeout=45)
(r/'usage_end.txt').write_text(u.stdout+u.stderr)
print(u.stdout)
