# Witness-VAD F1 / DSANet

F1 只检查单体结构、真实 B0 数据契约和梯度闭环，不做正式训练，也不看测试指标。

在远程仓库根目录执行：

~~~bash
source /etc/network_turbo
git pull
conda activate dsanet
bash run_instructions/run_witness_vad_f1_dsanet.sh
~~~

中断后复查：

~~~bash
bash run_instructions/run_witness_vad_f1_dsanet.sh --resume
~~~

只清理 F1 的精确输出目录并重跑：

~~~bash
bash run_instructions/run_witness_vad_f1_dsanet.sh --clean
~~~

结果位置：

~~~text
../vadmy_data/witness_vad/dsanet/f1_smoke/summary.md
../vadmy_data/witness_vad/dsanet/f1_smoke/test_report.txt
../vadmy_data/witness_vad/dsanet/f1_smoke/gradient_report.json
../vadmy_data/witness_vad/dsanet/f1_smoke/metrics.json
../vadmy_data/witness_vad/dsanet/f1_smoke/config.json
../vadmy_data/witness_vad/dsanet/f1_smoke/command.txt
../vadmy_data/witness_vad/dsanet/f1_smoke/stdout.log
~~~

判断很简单：`summary.md` 必须是 PASS；`test_report.txt` 不能有失败；
`gradient_report.json` 中五个 loss 的 `pass` 都必须为 true。否则不能进入 F2。
