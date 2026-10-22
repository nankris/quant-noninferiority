"""Collect run excerpts, normalizing internal identifiers for presentation only."""
from pathlib import Path
import json
r=Path('results');lines=['Quantization non-inferiority — session quant-noninferiority only','Saved nvidia-smi output:',(r/'nvidia_smi.txt').read_text(),'Session-created CLI history:',(r/'session_created.jsonl').read_text(),'Detached driver START/EXIT lines:']
lines.extend(x for x in (r/'driver.log').read_text().splitlines() if x.startswith(('START ','EXIT ')))
lines+=['Raw stage timing records:',(r/'stage_times.jsonl').read_text(),'Completion metadata excerpts (fields copied from *_done.json):']
for m in ['bf16','gptq','awq','hqq']:
 d=json.loads((r/f'{m}_done.json').read_text());p=json.loads((r/f'{m}_predictions.json').read_text())
 fields={k:d.get(k) for k in ['method','source_revision','quant_seconds','model_bytes','tokens_per_second','generation_seconds','eval_seconds','accuracy','finished_time','inference_linear_module_types']}
 lines+=[json.dumps(fields),f'Per-item checkpoint count ({m}): {len(p)}']
lines+=['CLI stop confirmation:',(r/'stop.txt').read_text(),'Final timing and CU estimation:',(r/'timing.json').read_text()]
text='\n'.join(lines)+'\n'
# Raw source files remain unchanged; only the displayed identifiers are normalized.
session=json.loads((r/'collection.json').read_text())['session']
experiment=json.loads((r/'timing.json').read_text())['experiment']
text=text.replace(session, 'quant-noninferiority').replace(experiment, 'quant-noninferiority')
Path('figures/colab_run.txt').write_text(text)
