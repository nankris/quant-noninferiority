"""Run detached on the dedicated VM; stage checkpoints survive driver restarts."""
import subprocess,time,pathlib,json,os,traceback
root=pathlib.Path('/content/quant-noninferiority');os.chdir(root);r=root/'results'
start=time.time();deadline=start+5.6*3600  # 8.624 CU at 1.54/h; independent watchdog includes setup.
for method in ['bf16','gptq','awq','hqq']:
    if (r/f'{method}_done.json').exists():continue
    meta=r/f'{method}_meta.json'
    checkpoint=meta.exists() and json.loads(meta.read_text()).get('model_bytes') and (root/'models'/method/'config.json').exists()
    stages=['eval'] if method=='bf16' or checkpoint else ['quantize','eval']
    for stage in stages:
        if (r/'STOP').exists():raise SystemExit('STOP requested')
        print('START',method,stage,time.time(),flush=True)
        env=dict(os.environ)
        if stage=='quantize':env['QUANT_ONLY']='1'
        else:env.pop('QUANT_ONLY',None)
        begin=time.time()
        with open(r/f'{method}_{stage}.log','a') as log:
            p=subprocess.run(['python','-u','run.py',method],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=max(1,deadline-time.time()))
        with open(r/'stage_times.jsonl','a') as f:f.write(json.dumps({'method':method,'stage':stage,'start':begin,'end':time.time(),'seconds':time.time()-begin,'returncode':p.returncode})+'\n')
        print('EXIT',method,stage,p.returncode,time.time(),flush=True)
        if p.returncode:break
(r/'driver_done.json').write_text(json.dumps({'finished':time.time()}))
