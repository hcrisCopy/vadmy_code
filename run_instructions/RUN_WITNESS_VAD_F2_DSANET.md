# Witness-VAD F2 / DSANet

F2 只检查训练、断点续跑和确定性。它使用 UCF 的训练清单，不读取测试清单或 GT，
也不把 smoke loss 当成性能结论。

在远程仓库根目录执行：

~~~bash
source /etc/network_turbo
git pull
conda activate dsanet
bash run_instructions/run_witness_vad_f2_dsanet.sh --clean
~~~

中断后继续或复查：

~~~bash
bash run_instructions/run_witness_vad_f2_dsanet.sh --resume
~~~

结果位置：

~~~text
../vadmy_data/witness_vad/dsanet/f2_train_contract/summary.md
../vadmy_data/witness_vad/dsanet/f2_train_contract/resume_report.json
../vadmy_data/witness_vad/dsanet/f2_train_contract/test_report.txt
../vadmy_data/witness_vad/dsanet/f2_train_contract/checkpoints/last.pt
../vadmy_data/witness_vad/dsanet/f2_train_contract/command.txt
../vadmy_data/witness_vad/dsanet/f2_train_contract/config.json
../vadmy_data/witness_vad/dsanet/f2_train_contract/metrics.json
../vadmy_data/witness_vad/dsanet/f2_train_contract/stdout.log
~~~

判断标准：`summary.md` 必须是 PASS；首轮 loss 相对误差不超过 1%；第 2 轮学习率
必须从第 1 轮结束值连续恢复；连续训练和续跑训练的最终参数误差不超过 `1e-7`。
