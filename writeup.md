# Can 4-bit Qwen3-8B stay within one accuracy point?

**On 15,214 questions, GPTQ lost 0.90 points (95% CI -1.35 to -0.45) and quantized in 5.33 minutes.**

GPTQ: **INCONCLUSIVE**; AWQ: **INCONCLUSIVE**; HQQ: **DISPROVEN**. These are separate method-level decisions under the preregistered rule. The headline recipe has the highest lower CI bound, with quantization time breaking ties; its interval is marginal and not adjusted for headline selection.

**Plain-English takeaway:** The full test sets did not establish a one-point accuracy guarantee for GPTQ or AWQ. HQQ quantized fastest, but its accuracy loss was clearly larger than one point.

| Recipe | Accuracy % [95% CI] | Change vs BF16, pp [95% CI] | Verdict | Quantize min | Saved GiB | Tokens/s |
|---|---:|---|---|---:|---:|---:|
| BF16 | 72.79 [72.07, 73.50] | reference | — | — | 15.26 | 15.0 |
| GPTQ | 71.89 [71.16, 72.60] | -0.90 [-1.35, -0.45] | INCONCLUSIVE | 5.33 | 5.70 | 8.1 |
| AWQ | 71.82 [71.09, 72.53] | -0.97 [-1.41, -0.55] | INCONCLUSIVE | 14.16 | 5.69 | 12.8 |
| HQQ | 70.36 [69.63, 71.09] | -2.43 [-2.91, -1.93] | DISPROVEN | 0.69 | 5.77 | 2.4 |

[Main results table](figures/table.png)

![Paired differences and one-point margin](figures/accuracy.png)

The environment used PyTorch 2.11.0+cu130, Transformers 5.17.0, GPTQModel 7.6.0, LLM Compressor 0.14.0 and HQQ 0.2.8.post1. I re-quantized the same Qwen/Qwen3-8B checkpoint (revision `b968826d9c46dd6066d109eabc6255188de91218`) using [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s recipes. All quantized models use 4-bit groups of 128, leaving vocabulary embeddings and the output head unquantized. GPTQ uses symmetric weights without activation ordering; AWQ uses W4A16_ASYM; HQQ uses axis 1 without calibration. GPTQ and AWQ share the same frozen 128 × 512-token WikiText-2 training calibration sequences from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq). All accuracy results and quantization timings here are new measurements; no predictions from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq) are pooled into them.

The primary outcome combines 14,042 MMLU test questions and 1,172 ARC-Challenge questions, weighted equally per item. Each model sees the same raw zero-shot prompt listing the question and choices and ending `Answer: `, with no chat template or generated reasoning. It scores the conditional log probability of the single-token answer letters. This reproduces [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s answer-label protocol; it does not score full answer text. Implementation deviation: direct batched Transformers calls replace the planned harness/vLLM evaluator to retain these native quantizer inference paths, as documented in README.md. Length-sorted batches use left padding, explicit positions and the final real token; a small batch/single-item validation is retained for each model.

For each recipe, the accuracy difference is quantized minus BF16, with 10,000 paired item bootstrap resamples (seed 44). The percentile 95% CI determines the decision: lower bound strictly above −1 point means PROVEN/non-inferior; upper bound strictly below −1 means DISPROVEN/inferior; otherwise INCONCLUSIVE. Decisions use unrounded bounds. An interval can exclude zero while crossing −1: evidence of some loss does not establish whether that loss exceeds the one-point margin. The bootstrap uses multinomial counts of the three possible paired differences, which has exactly the same resampling distribution as explicitly drawing item indices. Intervals are marginal, without a correction for testing three recipes; no multiplicity-controlled claim is made. GPTQ did not settle the one-point margin. AWQ did not settle the one-point margin. HQQ failed the one-point margin.

| Recipe | MMLU accuracy % [95% CI] | MMLU change pp [95% CI] | ARC accuracy % [95% CI] | ARC change pp [95% CI] |
|---|---|---|---|---|
| BF16 | 71.24 [70.47, 71.99] | reference | 91.38 [89.76, 92.92] | reference |
| GPTQ | 70.39 [69.63, 71.15] | -0.85 [-1.32, -0.37] | 89.85 [88.05, 91.55] | -1.54 [-2.82, -0.26] |
| AWQ | 70.32 [69.56, 71.08] | -0.92 [-1.37, -0.46] | 89.76 [87.97, 91.47] | -1.62 [-2.73, -0.51] |
| HQQ | 68.76 [67.98, 69.53] | -2.48 [-2.99, -1.96] | 89.59 [87.80, 91.30] | -1.79 [-3.07, -0.51] |

![Accuracy by full test set](figures/datasets.png)

HQQ quantized in 0.69 minutes, versus GPTQ 5.33 and AWQ 14.16. The secondary fastest-quantization hypothesis is descriptively **supported** in this one run; no timing CI is available. Quantization time starts before method-specific imports and cached loading and stops after quantization, excluding data/model downloads, checkpoint serialization, debugging and evaluation. File size is source safetensors for BF16 and the whole saved directory for quantized recipes, including small tokenizer/config files.

![Quantization time, disk footprint and throughput](figures/resources.png)

Generation throughput uses batch 1, a fixed 32-token prompt, exactly 128 greedy new tokens, one warmup and three timed repetitions with CUDA synchronization. These are measurements of the recorded library paths, not equally tuned production kernels. AWQ's compressed-tensors loader decompressed its saved 4-bit weights into ordinary BF16 Linear modules, with 16.54 billion resident parameter bytes (module classes are retained in `results/summary.json`). Its accuracy therefore evaluates reconstructed quantized weights, while its throughput describes this decompressed runtime. HQQ and GPTQ use their native PyTorch implementations. Model files preserve unquantized vocabulary matrices and metadata, so disk shrinkage need not equal fourfold.

The power check projected half-widths of ±0.33–0.43 points from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s paired disagreements, rather than promising them. Actual widths are in the main table. Expanding the sample improves precision but cannot force a non-inferiority verdict. The 92.3% MMLU weight in the combined score also differs from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s 50/50 dataset mix, so its combined accuracy loss is not directly comparable with the earlier headline.

Caveats: this is one model checkpoint, three recipe implementations and one calibration draw. An item bootstrap treats questions as independent and ignores subject/topic dependence; these full-test-set point estimates have no sampling error for the observed items, while the CIs express the chosen item-resampling inference model. Public benchmark contamination is not audited. Zero-shot answer-letter likelihood does not measure Qwen3 chat, long-context or thinking-mode performance. The batch-versus-single checks covered only three short items per method: all predictions agreed, but option log probabilities differed by as much as 0.44; this is a limited implementation check, not proof of arithmetic equivalence for every question. Timing order is fixed and each method was quantized only once. No quantization or evaluation stage failed in this run; console reconnect and archive-transfer issues are documented in README.md. No successful measurements are imported from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq).

Timing: estimated 135 minutes; actual 93.8 minutes; NVIDIA L4; 2.354 CU (session-time estimate at 1.54 CU/hour, not a session billing statement).

Exact environment versions, full prompts/data, every prediction and option log probability, batch checks, paired gains/losses, input hashes, stage timings and raw logs are in `results/`. `scripts/analyze.py` regenerates the tables, audit and 150-dpi charts offline. I used one session and stopped it after collection.
