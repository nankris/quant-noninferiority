"""Detached, incremental experiment. Run one method per process to release VRAM."""
import os, sys, time, json, math, random, traceback, pathlib, subprocess
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
ROOT=pathlib.Path('/content/quant-noninferiority'); R=ROOT/'results'; R.mkdir(exist_ok=True)
MODEL='Qwen/Qwen3-8B'
REVISION='b968826d9c46dd6066d109eabc6255188de91218'
torch.set_num_threads(4)
torch.manual_seed(44)
def dump(path,obj):
    tmp=pathlib.Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)
def prepare():
    if (R/'data.json').exists(): return json.loads((R/'data.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    frozen=json.loads((R/'calibration_source.json').read_text())
    m=load_dataset('cais/mmlu','all',split='test')
    a=load_dataset('allenai/ai2_arc','ARC-Challenge',split='test')
    items=[]
    for ds,name in [(m,'mmlu'),(a,'arc')]:
        ids=range(len(ds))
        for i in ids:
            row=ds[i]
            if name=='mmlu':
                choices=row['choices']; gold=int(row['answer']); q=row['question']
            else:
                choices=row['choices']['text']; gold=row['choices']['label'].index(row['answerKey']); q=row['question']
            items.append({'dataset':name,'source_index':i,'question':q,'choices':choices,'gold':gold,'subject':row.get('subject',name)})
    assert len(items)==15214, len(items)
    d={'items':items,'calibration_ids':frozen['calibration_ids'],'calibration_starts':frozen['calibration_starts'],'dataset_fingerprints':{'mmlu':m._fingerprint,'arc':a._fingerprint},'seed':44}
    dump(R/'data.json',d); tok.save_pretrained(ROOT/'tokenizer'); return d

def main(method):
    if (R/f'{method}_done.json').exists(): print('SKIP completed',method); return
    data=prepare()
    local=snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
    tok=AutoTokenizer.from_pretrained(local)
    meta_path=R/f'{method}_meta.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {'method':method,'model':MODEL,'source_revision':str(local).split('/')[-1],'calibration_sequences':0 if method in ['bf16','hqq'] else 128,'calibration_tokens_per_sequence':0 if method in ['bf16','hqq'] else 512}
    out=ROOT/'models'/method
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    if method=='bf16':
        model=AutoModelForCausalLM.from_pretrained(local,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        meta.update(quant_seconds=None,load_seconds=time.perf_counter()-start,quant_peak_gib=None,model_bytes=sum(p.stat().st_size for p in pathlib.Path(local).glob('*.safetensors')))
    elif method=='hqq':
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.core.quantize import BaseQuantizeConfig
        if (out/'qmodel.pt').exists() and meta.get('quant_seconds') and meta.get('model_bytes'):
            model=AutoHQQHFModel.from_quantized(str(out),compute_dtype=torch.bfloat16,device='cuda',attn_implementation='sdpa')
        else:
            model=AutoModelForCausalLM.from_pretrained(local,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
            AutoHQQHFModel.setup_model(model)
            qcfg=BaseQuantizeConfig(nbits=4,group_size=128,axis=1)
            configs={k:qcfg for k in model.linear_tags if 'lm_head' not in k}
            print('HQQ tags',model.linear_tags,'quantized tags',list(configs),flush=True)
            AutoHQQHFModel.quantize_model(model,quant_config=configs,compute_dtype=torch.bfloat16,device='cuda')
            torch.cuda.synchronize(); meta['quant_seconds']=time.perf_counter()-start; meta['quant_peak_gib']=torch.cuda.max_memory_allocated()/2**30
            meta['loader']='native hqq AutoHQQHFModel (Transformers 5 HqqConfig loader unsupported)'
            dump(meta_path,meta)
            out.mkdir(parents=True,exist_ok=True); AutoHQQHFModel.save_quantized(model,str(out)); tok.save_pretrained(out)
        meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    elif method=='gptq':
        from gptqmodel import GPTQModel, GPTQConfig, BACKEND
        if (out/'config.json').exists() and meta.get('quant_seconds') and meta.get('model_bytes'):
            wrapper=GPTQModel.load(str(out),device='cuda',backend=BACKEND.GPTQ_TORCH,dtype=torch.bfloat16)
        else:
            wrapper=GPTQModel.load(local,GPTQConfig(bits=4,group_size=128,desc_act=False,sym=True),dtype=torch.bfloat16)
            calib=[{'input_ids':x,'attention_mask':[1]*len(x)} for x in data['calibration_ids']]
            wrapper.quantize(calib,batch_size=1)
            torch.cuda.synchronize(); meta['quant_seconds']=time.perf_counter()-start;meta['quant_peak_gib']=torch.cuda.max_memory_allocated()/2**30
            dump(meta_path,meta); wrapper.save(str(out)); tok.save_pretrained(out)
            if os.environ.get('QUANT_ONLY')=='1':
                meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());dump(meta_path,meta);return
            # Offloaded quantization leaves meta tensors; reload the saved checkpoint for inference.
            del wrapper
            import gc;gc.collect();torch.cuda.empty_cache()
            wrapper=GPTQModel.load(str(out),device='cuda',backend=BACKEND.GPTQ_TORCH,dtype=torch.bfloat16)
        model=wrapper.model
        meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    elif method=='awq':
        from llmcompressor import oneshot
        from llmcompressor.modifiers.transform import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        if (out/'config.json').exists() and meta.get('quant_seconds') and meta.get('model_bytes'):
            model=AutoModelForCausalLM.from_pretrained(out,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        else:
            model=AutoModelForCausalLM.from_pretrained(local,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
            ds=Dataset.from_dict({'input_ids':data['calibration_ids'],'attention_mask':[[1]*512 for _ in range(128)]})
            oneshot(model=model,dataset=ds,recipe=[AWQModifier(),QuantizationModifier(ignore=['lm_head'],scheme='W4A16_ASYM',targets=['Linear'])],max_seq_length=512,num_calibration_samples=128)
            torch.cuda.synchronize(); meta['quant_seconds']=time.perf_counter()-start;meta['quant_peak_gib']=torch.cuda.max_memory_allocated()/2**30
            dump(meta_path,meta)
            model.save_pretrained(out,save_compressed=True);tok.save_pretrained(out)
            if os.environ.get('QUANT_ONLY')=='1':
                meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());dump(meta_path,meta);return
            # Reload the packed saved model, rather than evaluate fake-quant weights.
            del model
            import gc;gc.collect();torch.cuda.empty_cache()
            model=AutoModelForCausalLM.from_pretrained(out,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    model.eval(); meta['backend_modules']=sorted(set(type(m).__module__+'.'+type(m).__name__ for m in model.modules() if any(k in type(m).__name__.lower() for k in ['quant','hqq','compressed'])))
    dump(meta_path,meta)
    if os.environ.get('QUANT_ONLY')=='1':
        print('QUANTIZATION_CHECKPOINT_COMPLETE',method,flush=True);return
    torch.cuda.reset_peak_memory_stats(); eval_start=time.perf_counter()
    predpath=R/f'{method}_predictions.json'; preds=json.loads(predpath.read_text()) if predpath.exists() else []
    completed={x['item_id'] for x in preds}
    label_ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCDE']
    assert all(len(x)==1 for x in label_ids)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    encoded=[]
    for i,item in enumerate(data['items']):
        if i in completed:continue
        prompt='Choose the correct answer. Reply with only the letter.\n\n'+item['question']+'\n'+'\n'.join(f'{chr(65+j)}. {c}' for j,c in enumerate(item['choices']))+'\nAnswer: '
        encoded.append((i,tok.encode(prompt,add_special_tokens=False)))
    encoded.sort(key=lambda x:len(x[1]))
    token_budget=int(os.environ.get('TOKEN_BUDGET','2048')); max_batch=16
    batches=[];batch=[]
    for row in encoded:
        if batch and (len(batch)>=max_batch or len(row[1])*(len(batch)+1)>token_budget):
            batches.append(batch);batch=[]
        batch.append(row)
    if batch:batches.append(batch)
    print('EVAL_PLAN',method,'remaining',len(encoded),'batches',len(batches),'token_budget',token_budget,flush=True)
    with torch.inference_mode():
        for bi,batch in enumerate(batches):
            ids=tok.pad({'input_ids':[x[1] for x in batch]},padding=True,return_tensors='pt').to('cuda')
            pos=ids.attention_mask.cumsum(-1)-1;pos.masked_fill_(ids.attention_mask==0,0)
            logits=model(**ids,position_ids=pos,use_cache=False,logits_to_keep=1).logits[:,-1].float().log_softmax(-1)
            if bi==0 and not (R/f'{method}_batch_check.json').exists():
                checks=[]
                for k,(ci,ctokens) in enumerate(batch[:3]):
                    single=model(input_ids=torch.tensor([ctokens],device='cuda'),use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)
                    nchoices=len(data['items'][ci]['choices']);li=[v[0] for v in label_ids[:nchoices]]
                    sp=int(single[li].argmax());bp=int(logits[k,li].argmax())
                    checks.append({'item_id':ci,'single_prediction':sp,'batch_prediction':bp,'max_option_logprob_difference':float((single[li]-logits[k,li]).abs().max())})
                dump(R/f'{method}_batch_check.json',checks)
                assert all(c['single_prediction']==c['batch_prediction'] for c in checks),'Batch validation prediction disagreement'
            for k,(i,tokens) in enumerate(batch):
                item=data['items'][i]
                scores=[float(logits[k,x[0]]) for x in label_ids[:len(item['choices'])]]
                pred=max(range(len(scores)),key=lambda j:scores[j])
                preds.append({'item_id':i,'dataset':item['dataset'],'source_index':item['source_index'],'subject':item['subject'],'gold':item['gold'],'prediction':pred,'correct':int(pred==item['gold']),'scores':scores,'prompt_tokens':len(tokens)})
            dump(predpath,preds)
            if bi%10==0:
                elapsed=time.perf_counter()-eval_start
                print(method,'ACCURACY',len(preds),'elapsed',round(elapsed,2),'items_per_second',round((len(preds)-len(completed))/elapsed,3),flush=True)
            if (R/'STOP').exists():raise RuntimeError('Budget watchdog requested stop')
        # Exactly 32 input tokens; fixed prompt repeated if needed, then sliced.
        ids=tok('Explain how computers represent numbers. '*10,return_tensors='pt').input_ids[:,:32].to('cuda')
        genkw={'max_new_tokens':128,'min_new_tokens':128,'do_sample':False,'pad_token_id':tok.eos_token_id,'use_cache':True}
        model.generate(ids,**genkw);torch.cuda.synchronize(); times=[]
        for _ in range(3):
            t=time.perf_counter(); result=model.generate(ids,**genkw);torch.cuda.synchronize(); times.append(time.perf_counter()-t)
        meta.update(tokens_per_second=128/(sum(times)/len(times)),generation_seconds=times,eval_peak_gib=torch.cuda.max_memory_allocated()/2**30,eval_seconds=time.perf_counter()-eval_start,accuracy=sum(x['correct'] for x in preds)/len(preds),finished_time=time.time())
    meta['resident_parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
    meta['inference_linear_module_types']=sorted(set(type(m).__module__+'.'+type(m).__name__ for m in model.modules() if 'linear' in type(m).__name__.lower()))
    dump(meta_path,meta);dump(R/f'{method}_done.json',meta)
    print('DONE',method,json.dumps(meta),flush=True)
if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare': prepare()
        else: main(sys.argv[1])
    except Exception:
        err=traceback.format_exc();print(err,flush=True);dump(R/f'{sys.argv[1]}_error.json',{'time':time.time(),'error':err});raise
