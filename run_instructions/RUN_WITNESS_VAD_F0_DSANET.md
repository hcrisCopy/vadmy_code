# WITNESS-VAD / DSANet：F0 Universal 提点验尸

本阶段不训练新模型，只在同一批历史缓存上关闭不同信息来源。目标是确认
Universal 为什么涨点，避免 v10 把真正有效的信息删掉。

这里复现的是历史 Universal，因此沿用它原始的 seed 234；从 F1 开始，
WITNESS-VAD 的所有新模块统一使用 seed 42。

## 1. 登录并更新

~~~bash
ssh -p 19423 root@connect.cqa1.seetacloud.com
cd /root/autodl-tmp/vadmy_code
source /etc/network_turbo
git pull origin main
conda activate dsanet
~~~

## 2. 正式运行

~~~bash
bash run_instructions/run_witness_vad_f0_dsanet.sh
~~~

中断后续跑：

~~~bash
bash run_instructions/run_witness_vad_f0_dsanet.sh --resume
~~~

只清理并重跑 F0：

~~~bash
bash run_instructions/run_witness_vad_f0_dsanet.sh --clean
~~~

三条命令和脚本实际接受的命令完全一致。默认运行也会复用已经完成且输入哈希一致的变体。

## 3. 看结果

~~~bash
cat ../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/summary.md
cat ../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/information_source.json
column -s, -t < ../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/comparison.csv | less -S
~~~

重点看：

- historical_reproduction_abs_error_pp：必须不超过 0.1 pp；
- component_drop_from_full_pp：删除某个信息来源后掉多少；
- dominant_source：主要来自视频级 suppression、局部神经元校正，还是组合；
- Cross-AUC、Macro-Within-AUC、异常视频指标和 Normal FPR：判断涨点究竟改变了什么。

产物都在 ../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/：

| 产物 | 用途 |
|---|---|
| summary.md | 人话结论 |
| comparison.csv | U0--U4 完整指标 |
| information_source.json | 信息来源裁决 |
| source_audit.json | 输入路径、行数与哈希 |
| config.json | 正式参数 |
| command.txt | 实际执行命令 |
| stdout.log | 完整日志 |
| dataset/variant/ | 每个变体的曲线、原始指标和诊断指标 |

## 一键运行

~~~bash
bash run_instructions/run_witness_vad_f0_dsanet.sh
~~~
