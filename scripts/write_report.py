from pathlib import Path
import json
r=Path('results');s=json.loads((r/'summary.json').read_text());t=json.loads((r/'timing.json').read_text());d=s['methods'];best=s['headline_method'];names={'bf16':'BF16','gptq':'GPTQ','awq':'AWQ','hqq':'HQQ'}
def delta(row):
 lo,hi=row['delta_ci'];return f"{row['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
def acc(row):
 lo,hi=row['accuracy_ci'];return f"{row['accuracy_pct']:.2f} [{lo:.2f}, {hi:.2f}]"
def table():
 lines=['| Recipe | Accuracy % [95% CI] | Change vs BF16, pp [95% CI] | Verdict | Quantize min | Saved GiB | Tokens/s |','|---|---:|---|---|---:|---:|---:|']
 for m in names:
  z=d[m];lines.append(f"| {names[m]} | {acc(z)} | {'reference' if m=='bf16' else delta(z)} | {z.get('verdict','—')} | {'—' if m=='bf16' else format(z['quant_seconds']/60,'.2f')} | {z['model_bytes']/2**30:.2f} | {z['tokens_per_second']:.1f} |")
 return '\n'.join(lines)
b=d[best];lo,hi=b['delta_ci'];loss=-b['delta_pp'];direction='lost' if loss>=0 else 'gained'
headline=f"On 15,214 questions, {names[best]} {direction} {abs(loss):.2f} points (95% CI {lo:+.2f} to {hi:+.2f}) and quantized in {b['quant_seconds']/60:.2f} minutes."
verdicts='; '.join(f'{names[m]}: **{d[m]["verdict"]}**' for m in ['gptq','awq','hqq'])
timing=f"Timing: estimated {t['estimated_minutes']['total']} minutes; actual {t.get('actual_minutes',0):.1f} minutes; NVIDIA L4; {t.get('cu_used_estimate',0):.3f} CU (session-time estimate at 1.54 CU/hour, not a session billing statement)."
verdict_sentence=' '.join(f"{names[m]} {'met' if d[m]['verdict']=='PROVEN' else 'failed' if d[m]['verdict']=='DISPROVEN' else 'did not settle'} the one-point margin." for m in ['gptq','awq','hqq'])
text=f'''# Can 4-bit Qwen3-8B stay within one accuracy point?

**{headline}**

{verdicts}. These are separate method-level decisions under the preregistered rule. The headline recipe has the highest lower CI bound, with quantization time breaking ties; its interval is marginal and not adjusted for headline selection.

**Plain-English takeaway:** The full test sets did not establish a one-point accuracy guarantee for GPTQ or AWQ. HQQ quantized fastest, but its accuracy loss was clearly larger than one point.

{table()}

[Main results table](figures/table.png)

![Paired differences and one-point margin](figures/accuracy.png)

The environment used PyTorch 2.11.0+cu130, Transformers 5.17.0, GPTQModel 7.6.0, LLM Compressor 0.14.0 and HQQ 0.2.8.post1. I re-quantized the same Qwen/Qwen3-8B checkpoint (revision `b968826d9c46dd6066d109eabc6255188de91218`) using [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s recipes. All quantized models use 4-bit groups of 128, leaving vocabulary embeddings and the output head unquantized. GPTQ uses symmetric weights without activation ordering; AWQ uses W4A16_ASYM; HQQ uses axis 1 without calibration. GPTQ and AWQ share the same frozen 128 × 512-token WikiText-2 training calibration sequences from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq). All accuracy results and quantization timings here are new measurements; no predictions from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq) are pooled into them.

The primary outcome combines 14,042 MMLU test questions and 1,172 ARC-Challenge questions, weighted equally per item. Each model sees the same raw zero-shot prompt listing the question and choices and ending `Answer: `, with no chat template or generated reasoning. It scores the conditional log probability of the single-token answer letters. This reproduces [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s answer-label protocol; it does not score full answer text. Implementation deviation: direct batched Transformers calls replace the planned harness/vLLM evaluator to retain these native quantizer inference paths, as documented in README.md. Length-sorted batches use left padding, explicit positions and the final real token; a small batch/single-item validation is retained for each model.

For each recipe, the accuracy difference is quantized minus BF16, with 10,000 paired item bootstrap resamples (seed 44). The percentile 95% CI determines the decision: lower bound strictly above −1 point means PROVEN/non-inferior; upper bound strictly below −1 means DISPROVEN/inferior; otherwise INCONCLUSIVE. Decisions use unrounded bounds. An interval can exclude zero while crossing −1: evidence of some loss does not establish whether that loss exceeds the one-point margin. The bootstrap uses multinomial counts of the three possible paired differences, which has exactly the same resampling distribution as explicitly drawing item indices. Intervals are marginal, without a correction for testing three recipes; no multiplicity-controlled claim is made. {verdict_sentence}

| Recipe | MMLU accuracy % [95% CI] | MMLU change pp [95% CI] | ARC accuracy % [95% CI] | ARC change pp [95% CI] |
|---|---|---|---|---|
'''
for m in names:
 a=s['datasets']['mmlu'][m];c=s['datasets']['arc'][m]
 text+=f"| {names[m]} | {acc(a)} | {'reference' if m=='bf16' else delta(a)} | {acc(c)} | {'reference' if m=='bf16' else delta(c)} |\n"
text+=f'''
![Accuracy by full test set](figures/datasets.png)

HQQ quantized in {d['hqq']['quant_seconds']/60:.2f} minutes, versus GPTQ {d['gptq']['quant_seconds']/60:.2f} and AWQ {d['awq']['quant_seconds']/60:.2f}. The secondary fastest-quantization hypothesis is descriptively **{'supported' if s['hqq_fastest'] else 'not supported'}** in this one run; no timing CI is available. Quantization time starts before method-specific imports and cached loading and stops after quantization, excluding data/model downloads, checkpoint serialization, debugging and evaluation. File size is source safetensors for BF16 and the whole saved directory for quantized recipes, including small tokenizer/config files.

![Quantization time, disk footprint and throughput](figures/resources.png)

Generation throughput uses batch 1, a fixed 32-token prompt, exactly 128 greedy new tokens, one warmup and three timed repetitions with CUDA synchronization. These are measurements of the recorded library paths, not equally tuned production kernels. AWQ's compressed-tensors loader decompressed its saved 4-bit weights into ordinary BF16 Linear modules, with {d['awq']['resident_parameter_bytes']/1e9:.2f} billion resident parameter bytes (module classes are retained in `results/summary.json`). Its accuracy therefore evaluates reconstructed quantized weights, while its throughput describes this decompressed runtime. HQQ and GPTQ use their native PyTorch implementations. Model files preserve unquantized vocabulary matrices and metadata, so disk shrinkage need not equal fourfold.

The power check projected half-widths of ±0.33–0.43 points from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s paired disagreements, rather than promising them. Actual widths are in the main table. Expanding the sample improves precision but cannot force a non-inferiority verdict. The 92.3% MMLU weight in the combined score also differs from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s 50/50 dataset mix, so its combined accuracy loss is not directly comparable with the earlier headline.

Caveats: this is one model checkpoint, three recipe implementations and one calibration draw. An item bootstrap treats questions as independent and ignores subject/topic dependence; these full-test-set point estimates have no sampling error for the observed items, while the CIs express the chosen item-resampling inference model. Public benchmark contamination is not audited. Zero-shot answer-letter likelihood does not measure Qwen3 chat, long-context or thinking-mode performance. The batch-versus-single checks covered only three short items per method: all predictions agreed, but option log probabilities differed by as much as 0.44; this is a limited implementation check, not proof of arithmetic equivalence for every question. Timing order is fixed and each method was quantized only once. No quantization or evaluation stage failed in this run; console reconnect and archive-transfer issues are documented in README.md. No successful measurements are imported from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq).

{timing}

Exact environment versions, full prompts/data, every prediction and option log probability, batch checks, paired gains/losses, input hashes, stage timings and raw logs are in `results/`. `scripts/analyze.py` regenerates the tables, audit and 150-dpi charts offline. I used one session and stopped it after collection.
'''
Path('writeup.md').write_text(text)
p=Path('README.md');readme=p.read_text();marker='\n## Completed results\n';readme=readme.split(marker)[0]
readme+=f'\n## Completed results\n\n{headline}\n\n{verdicts}.\n\n{timing}\n\nSee writeup.md, results/summary.json and results/audit.json. Full coverage: 15,214 unique paired items per model, 60,856 predictions total. No dataset reduction. The named session is stopped; evidence is in results/stop.txt.\n'
p.write_text(readme)
print(headline)
