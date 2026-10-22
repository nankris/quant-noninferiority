import pathlib,subprocess,tarfile,json,time,re,shutil,io
root=pathlib.Path('/content/quant-noninferiority');r=root/'results'
print(subprocess.getoutput("ps -eo pid,etime,args | grep -E 'driver.py|run.py|install|watchdog' | grep -v grep"))
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'))
for method in ['bf16','gptq','awq','hqq']:
 done=r/f'{method}_done.json'
 if done.exists():
  d=json.loads(done.read_text());print('COMPLETE',method,'accuracy',round(d['accuracy']*100,4),'quant_seconds',d.get('quant_seconds'));continue
 for stage in ['quantize','eval']:
  p=r/f'{method}_{stage}.log'
  if p.exists():
   text=p.read_text(errors='replace');layers=re.findall(r'HessianInputDedup:.*',text)
   print(method,stage,'\n',layers[-1] if layers else '\n'.join(text.splitlines()[-2:])[-700:])
if pathlib.Path('/content/bootstrap-quant.log').exists():shutil.copyfile('/content/bootstrap-quant.log',r/'driver.log')
for p in (root/'models').glob('*/*.json'):
 if p.stat().st_size<1000000:
  target=r/'model_configs'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
(r/'collection.json').write_text(json.dumps({'collected_unix':time.time(),'session':'quant-noninferiority'}))
with tarfile.open(root/'checkpoint.tar.gz.tmp','w:gz') as tar:
 for p in r.rglob('*'):
  if p.is_file() and not p.name.endswith('.tmp'):
   payload=p.read_bytes();info=tarfile.TarInfo('results/'+str(p.relative_to(r)));info.size=len(payload);info.mtime=time.time();tar.addfile(info,io.BytesIO(payload))
(root/'checkpoint.tar.gz.tmp').replace(root/'checkpoint.tar.gz')
print('CHECKPOINT_READY',time.time())
