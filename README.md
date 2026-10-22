# Can 4-bit Qwen3-8B stay within one accuracy point?

Question: Which of GPTQ, AWQ and HQQ are non-inferior to BF16 on the full MMLU and ARC-Challenge tests, with a one-percentage-point margin? Hypothesis: each method is non-inferior; secondary hypothesis: HQQ quantizes fastest without calibration. This is a new run, not an extension of [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s 400 predictions.

## Preregistration (before GPU evaluation)

Model: Qwen/Qwen3-8B, revision b968826d9c46dd6066d109eabc6255188de91218. Four-bit groups of 128; GPTQ symmetric, desc_act=False; AWQ W4A16_ASYM; native HQQ axis=1. Embeddings and output head are unquantized. Reuse [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s frozen 128 WikiText-2 train calibration sequences of 512 tokens for GPTQ/AWQ; HQQ uses none. Re-quantize all three methods and measure wall time from method imports/cached model loading through quantization, excluding download, serialization and debugging. No old predictions or quantized weights enter new estimates.

Primary sample: all 14,042 `cais/mmlu` all/test questions plus all 1,172 `allenai/ai2_arc` ARC-Challenge/test questions: 15,214 paired items. Combined accuracy is item-weighted. Also report each dataset separately. All four methods receive identical raw zero-shot prompts from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq), ending `Answer: `, and score the conditional log likelihood of each single-token answer letter. This is answer-label likelihood, not full answer-text likelihood, and uses no chat template/thinking. Batching and length sorting may accelerate inference; they do not change prompts. Save every item, gold answer, prediction and option score.

For each method independently, calculate quantized minus BF16 accuracy in percentage points, with 10,000 paired item bootstrap resamples, seed 44, percentile 95% CI. PROVEN (NON-INFERIOR) if lower bound > -1.0; DISPROVEN (INFERIOR) if upper bound < -1.0; otherwise INCONCLUSIVE. There is no requirement that all three pass and no familywise correction: these are three marginal method-level claims. The primary verdict is based on combined items; dataset breakdowns are secondary. Select the headline method by highest primary lower confidence bound (ties: smaller measured quantization time), regardless of its verdict. Missing methods remain inconclusive. The 95% two-sided interval rule is conservative relative to a one-sided 5% non-inferiority test.

Power check: for paired differences d in {-1,0,1}, expected half-width is 1.96 * 100 * sqrt((p_discordant - mean(d)^2)/n). At n=15,214 and 6% discordance, this is 0.389 percentage points, about ±0.4; at 10% it is ±0.50, and the worst-case bound is ±1.59. The ±0.4 estimate is conditional, not guaranteed. Observed discordances from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq) are recorded in results/power_check.json before inference. Narrower CIs can settle genuine losses as INFERIOR; more questions do not guarantee NON-INFERIORITY.

Secondary measures: single-run quantization seconds, saved directory bytes (BF16 source safetensors only), batch-1 greedy generation tokens/s using [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s 32-input/128-output-token prompt, one warmup and three repetitions. No timing confidence interval. Backend implementations are retained from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq); AWQ may decompress into BF16 Linear modules, and HQQ/GPTQ may use unoptimized PyTorch paths. Record actual module classes; throughput compares these implementations. Perplexity is omitted because it is outside this experiment's primary question.

Budget: L4, session quant-noninferiority only, cap 10 CU. Use detached GPU jobs and incremental per-batch atomic checkpoints. Estimate full evaluation from early timing; if it cannot fit the remaining budget with a safety reserve, remove ARC first, then use a deterministic subject-stratified 6,000 MMLU items, documenting the change here before that reduced evaluation. Never silently pool unequal samples. A watchdog ends work before the cap. The session stops on completion or an unrecoverable failure.

Timing plan: estimated 135 minutes (setup 20, quantization/evaluation 100, writeup 15); GPU L4 planned; Actual timing and resource accounting appear below. Start timestamp is in results/timing.json.

## Reproducibility and artifacts

The original comparison supplied frozen calibration inputs; its files were not modified. scripts/ contains the adapted GPU runner, detached driver, synchronization and analysis. results/ stores frozen data, predictions, timing, environment and logs. figures/ contains publication PNGs and main table image. Large model weights remain on the dedicated VM during the run and are reproducible from the pinned source.

Execution note (before model evaluation): the assigned VM's Jupyter execution channel repeatedly failed with HTTP 503 and `Connection was lost`, including after a kernel restart. Its terminal websocket works. Detached jobs and checkpoint collection therefore use `colab console -s quant-noninferiority` plus `colab upload/download`, rather than relying on broken Jupyter channels. This does not change the inference protocol. No other session was accessed.

Pilot power projections from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq): GPTQ ±0.412 points (6.75% discordance), AWQ ±0.326 (4.25%), HQQ ±0.433 (7.50%). These are planning calculations, not results of this run.

Evaluator implementation deviation, recorded before accuracy results: evaluation uses an adapted, length-batched Transformers scorer from [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq) rather than lm-eval-harness/vLLM. This keeps the native HQQ and GPTQ Torch inference paths and [my original GPTQ/AWQ/HQQ comparison](https://github.com/nankris/gptq-awq-hqq)'s exact answer-label prompts consistent across all four models; no vLLM compatibility or throughput advantage is assumed. A single causal forward computes conditional log probabilities for all answer labels, so no generated answers or text matching enter correctness. This remains a log-likelihood multiple-choice test. Initial full data preparation confirmed all 15,214 items; no sample reduction has been invoked.

Runtime budget backstop: a detached watchdog on this VM is scheduled for 2026-10-17 23:00:30 UTC, 5.75 hours after the CLI session-created event (8.855 CU at 1.54 CU/hour, leaving >1 CU safety margin). It writes STOP/checkpoints and asks only its own runtime management service to unassign the VM. Normal completion still uses `colab stop -s quant-noninferiority` after the final local sync. The watchdog runs independently of the local connection.

## Recovery and reproduction notes

The original detached driver completed fresh quantization and evaluation stages without a method failure. A console reconnect once dropped its immediate command, and a later download caught an archive being written; both were detected. `scripts/poll.py` now atomically publishes archives, while `scripts/sync.sh` verifies gzip integrity and an embedded collection timestamp before accepting a checkpoint. Neither problem required rerunning any GPU measurement.

Offline reproduction uses `python scripts/analyze.py` followed by `scripts/write_report.py`; completed timing is already saved. `scripts/finalize.py` records the wall time at the moment it is invoked, so only rerun it when intentionally updating finalization timing. `scripts/proof.py` rebuilds the terminal evidence file. These scripts require numpy and matplotlib, and analysis validates all predictions before generating figures.

A fresh GPU run needs the recorded packages from scripts/install.sh, the pinned source model, scripts/run.py, scripts/driver.py, scripts/watchdog.py, scripts/poll.py and frozen results/data.json in the remote experiment directory (`quant-noninferiority` under the Colab working directory). Keep the original calibration IDs and prompts. For a fresh allocation, reset the watchdog deadline from that allocation time rather than reusing the historical deadline in results/budget.json. Start the driver with `nohup python -u driver.py > results/driver.log 2>&1 < /dev/null &`, confirm it is not already running before any restart, and synchronize often. Always inspect `colab sessions` and `colab usage` before allocating this named session. scripts/stop.py checks local completion before stopping only quant-noninferiority. Large saved weights are not included in the repository; regenerate them from the pinned source and frozen calibration inputs.

## Completed results

On 15,214 questions, GPTQ lost 0.90 points (95% CI -1.35 to -0.45) and quantized in 5.33 minutes.

GPTQ: **INCONCLUSIVE**; AWQ: **INCONCLUSIVE**; HQQ: **DISPROVEN**.

Timing: estimated 135 minutes; actual 93.8 minutes; NVIDIA L4; 2.354 CU (session-time estimate at 1.54 CU/hour, not a session billing statement).

See writeup.md, results/summary.json and results/audit.json. Full coverage: 15,214 unique paired items per model, 60,856 predictions total. No dataset reduction. The named session is stopped; evidence is in results/stop.txt.
