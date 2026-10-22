"""Detached budget backstop on this Colab VM; affects only its own runtime."""
import os,time,json,pathlib,urllib.request,tarfile
root=pathlib.Path('/content/quant-noninferiority');r=root/'results'
cfg=json.loads((root/'budget.json').read_text());deadline=cfg['deadline_unix']
(r/'watchdog_started.json').write_text(json.dumps({'started':time.time(),'deadline':deadline,'max_planned_cu':8.855,'runtime_management_service_available':bool(os.environ.get('TBE_RUNTIME_ADDR'))},indent=2))
print('WATCHDOG_READY',deadline,flush=True)
while time.time()<deadline:time.sleep(min(30,max(.1,deadline-time.time())))
(r/'STOP').write_text('Budget backstop reached')
(r/'watchdog_fired.json').write_text(json.dumps({'time':time.time()}))
with tarfile.open(root/'checkpoint.tar.gz','w:gz') as t:
 for p in r.rglob('*'):
  if p.is_file() and not p.name.endswith('.tmp'):t.add(p,arcname='results/'+str(p.relative_to(r)))
addr=os.environ.get('TBE_RUNTIME_ADDR')
if not addr:raise RuntimeError('No runtime management address; cannot enforce VM unassignment')
print('UNASSIGN_OWN_RUNTIME',flush=True)
req=urllib.request.Request(f'http://{addr}/unassign',method='POST')
with urllib.request.urlopen(req,timeout=20) as resp:print('UNASSIGN_RESPONSE',resp.status,flush=True)
