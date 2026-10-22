"""Finalize wall-clock accounting after stopping quant-noninferiority."""
import json,datetime,pathlib,time
r=pathlib.Path('results');t=json.loads((r/'timing.json').read_text());e=json.loads((r/'resource_end.json').read_text())
start=datetime.datetime.fromisoformat(t['start_ist']).timestamp();end=time.time();stop=e['stop_unix'];stages=[json.loads(x) for x in (r/'stage_times.jsonl').read_text().splitlines() if x.strip()]
quant=sum(x['seconds'] for x in stages if x['stage']=='quantize');ev=sum(x['seconds'] for x in stages if x['stage']=='eval')
# First BF16 stage includes public dataset/model preparation; classify that overhead as setup.
b=json.loads((r/'bf16_done.json').read_text());bs=next(x for x in stages if x['method']=='bf16' and x['stage']=='eval')
prep=max(0,bs['seconds']-b['eval_seconds']-b['load_seconds']);ev-=prep
setup=max(0,stop-start-quant-ev);writeup=end-stop
allocation=e.get('allocation_unix',start)
cu=(stop-allocation)/3600*1.54
ist=datetime.timezone(datetime.timedelta(hours=5,minutes=30))
t.update(end_ist=datetime.datetime.fromtimestamp(end,ist).isoformat(),actual_minutes=(end-start)/60,per_stage_minutes={'setup':setup/60,'gpu_run':quant/60,'eval':ev/60,'writeup':writeup/60},stage_definitions='Exclusive wall-clock buckets: setup includes preparation and runtime overhead; GPU run means quantization stages including serialization; eval includes model loading; writeup is finalization after GPU stop. Script/report preparation also overlapped GPU work.',gpu='NVIDIA L4 (23,034 MiB)',cu_used_estimate=cu,cu_accounting='session allocation-to-stop duration times 1.54 CU/hour; not a per-session bill',gpu_stop_ist=datetime.datetime.fromtimestamp(stop,ist).isoformat(),allocation_unix=allocation)
(r/'timing.json').write_text(json.dumps(t,indent=2));assert cu<10,cu
print(json.dumps(t,indent=2))
