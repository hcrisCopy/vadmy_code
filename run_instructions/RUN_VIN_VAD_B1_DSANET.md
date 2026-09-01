# DSANet 正式流程：B1 正常上下文预测器

B1 只验证一件事：不看目标附近 hidden 的情况下，正常上下文能否比逐神经元全局均值更准确地预测目标响应。这里不计算检测 AUC/AP。

## 1. 登录和更新

```bash
ssh -p 19423 root@connect.cqa1.seetacloud.com
cd /root/autodl-tmp/vadmy_code
source /etc/network_turbo
git pull
conda activate dsanet
```

## 2. 正式运行

```bash
bash run_instructions/run_vin_vad_b1_dsanet.sh
```

正式配置使用 batch size 8。训练时每个正常视频每轮抽取一个由
`seed + epoch + video key` 决定的可复现窗口；每个 held-out 正常验证视频
使用一个固定中心窗口（最多 256 个目标片段）。这样避免重复解压同一视频，
同时保持视频级独立划分和固定验收口径。

中断后执行同一条命令，会从最近一个 epoch 继续。需要清空 B1 后重跑：

```bash
CLEAN=1 bash run_instructions/run_vin_vad_b1_dsanet.sh
```

## 3. 看结果

```bash
cat ../vadmy_data/vin_vad/dsanet/b1/ucf/summary.json
cat ../vadmy_data/vin_vad/dsanet/b1/xd/summary.json
```

`validation_conditional_nll` 必须小于 `validation_global_nll`。否则说明上下文预测没有提供增量信息，B1 失败，不能进入 B2。

产物：

| 产物 | 相对位置 |
|---|---|
| 完整日志 | `../vadmy_data/vin_vad/dsanet/b1/run.log` |
| normal-only 划分 | `../vadmy_data/vin_vad/dsanet/b1/<dataset>/data/` |
| 全局高斯基线 | `../vadmy_data/vin_vad/dsanet/b1/<dataset>/global_normal_statistics.npz` |
| 最优预测器 | `../vadmy_data/vin_vad/dsanet/b1/<dataset>/context_predictor_best.pt` |
| 每 epoch 记录 | `../vadmy_data/vin_vad/dsanet/b1/<dataset>/history.json` |
| 最终验收 | `../vadmy_data/vin_vad/dsanet/b1/<dataset>/summary.json` |

## 一键全部运行

```bash
bash run_instructions/run_vin_vad_b1_dsanet.sh
```
