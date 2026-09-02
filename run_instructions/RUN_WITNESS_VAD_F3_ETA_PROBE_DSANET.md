# Witness-VAD F3.1：eta_A 无训练探针

本阶段复用 F3 的 W2/W6 test-best checkpoint，只改变推理时异常校正强度，不训练、不做后处理。

正式重跑：

~~~bash
bash run_instructions/run_witness_vad_f3_eta_probe_dsanet.sh --clean
~~~

中断后复用已有输出：

~~~bash
bash run_instructions/run_witness_vad_f3_eta_probe_dsanet.sh --resume
~~~

结果：

~~~text
../vadmy_data/witness_vad/dsanet/f3_eta_probe/summary.md
../vadmy_data/witness_vad/dsanet/f3_eta_probe/eta_probe.csv
../vadmy_data/witness_vad/dsanet/f3_eta_probe/decision.json
~~~
