# Run record

I froze the protocol and power check before evaluation (commit `4fe327c`). The full sample was retained. All seven GPU stages returned zero; all four model measurements and 60,856 predictions were collected and audited. The session stopped after collection, as recorded in `stop.txt`.

Milestone at ~17:45 UTC: BF16 completed all 15,214 items (72.7882%; 10,003/14,042 MMLU and 1,071/1,172 ARC). Fresh GPTQ quantization completed in 319.770686331 s; saved model metadata includes model_bytes, configs retained under results/model_configs/gptq. GPTQ evaluation was then active. Installer completed successfully. Environment: Torch 2.11.0+cu130, Transformers 5.17.0, GPTQModel 7.6.0, LLM Compressor 0.14.0, HQQ 0.2.8.post1.

Milestone 17:56 UTC: full GPTQ evaluation done (71.8877%; delta −0.900486 pp, paired 95% CI [−1.354016, −0.446957], INCONCLUSIVE). Saved results/gptq_paired.json. The measured effect is close to the frozen margin. AWQ quantization is now running, then its full eval, then HQQ. Both completed methods have 15,214 predictions. All full data retained.

Milestone 18:23 UTC: AWQ full evaluation complete: 71.8154%, delta −0.972788 pp [−1.406599, −0.545550], INCONCLUSIVE. HQQ quantization active under the original detached driver. AWQ observed ordinary BF16 Linear modules and 16.54B resident parameter bytes. Console sync now waits five seconds before sending commands, because an earlier immediate write was dropped and yielded a stale archive; the next fresh poll recovered correctly. No GPU stage was relaunched.

Milestone 18:29 UTC: HQQ re-quantized successfully in 41.232683 s, saved 6,191,602,118 bytes, native HQQLinear backend; evaluation past 7,300 items, no errors. Fresh checkpoint marker results/collection.json now lets sync reject stale archives. Full evaluation and 4 × 128-token generation runs were still pending.

Milestone 18:35 UTC: HQQ past 11,800 items. A truncated archive download was detected by gzip; no GPU process failed. Remote poll now builds checkpoint.tar.gz.tmp then atomically renames it; sync downloads to a temporary name and gzip-validates before extraction, then checks collection timestamp. Subsequent full archive restored all local JSONs successfully. At this milestone, scripts/stop.py had not yet been invoked: it requires all four done records and full predictions, then stops only the experiment session and records resource_end.json/stop.txt/usage_end.txt.

Milestone 18:43 UTC: All HQQ predictions collected: delta −2.425398 pp, paired 95% CI [−2.905219, −1.932431], DISPROVEN. Accuracy complete for all four; HQQ batch-1 generation warmup + three repetitions is still active. Completion records were collected after generation finished. Primary headline GPTQ (highest lower CI): −0.900486 pp, CI [−1.354016, −0.446957], quantization 5.329511 minutes. GPTQ/AWQ INCONCLUSIVE; HQQ INFERIOR.

Primary verdicts: GPTQ INCONCLUSIVE; AWQ INCONCLUSIVE; HQQ DISPROVEN. HQQ was the fastest quantizer, at 41.23 seconds.
