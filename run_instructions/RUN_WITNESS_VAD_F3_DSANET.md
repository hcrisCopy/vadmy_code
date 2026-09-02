# Witness-VAD F3 / DSANet

F3 是性能生死门：DSANet 上固定跑 W0、W1、W2、W6，UCF/XD 使用同一套超参数，
统一读取最后一个 checkpoint。这里不做 test-best epoch，也不做后处理。

首次正式运行：

~~~bash
source /etc/network_turbo
git pull
conda activate dsanet
bash run_instructions/run_witness_vad_f3_dsanet.sh --clean
~~~

SSH 中断后继续：

~~~bash
bash run_instructions/run_witness_vad_f3_dsanet.sh --resume
~~~

主要结果：

~~~text
../vadmy_data/witness_vad/dsanet/f3_performance/summary.md
../vadmy_data/witness_vad/dsanet/f3_performance/main_results.csv
../vadmy_data/witness_vad/dsanet/f3_performance/error_decomposition.json
../vadmy_data/witness_vad/dsanet/f3_performance/test_report.txt
../vadmy_data/witness_vad/dsanet/f3_performance/stdout.log
~~~

每个训练的最后 checkpoint 在：

~~~text
../vadmy_data/witness_vad/dsanet/f3_performance/<dataset>/<variant>/training/checkpoints/last.pt
~~~

`summary.md` 会直接写 GO 或 NO-GO。NO-GO 时脚本返回失败并停止，不进入 F4。
