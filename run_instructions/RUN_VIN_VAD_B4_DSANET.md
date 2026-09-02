# DSANet 正式流程：B4 单次联合训练

B4 把 B1 predictor、B2 directional field 和 B3 两轴 auditor 接成一个训练图。
本阶段只验收训练、梯度隔离、正常统计和完整 checkpoint，不看测试 AUC/AP。

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
bash run_instructions/run_vin_vad_b4_dsanet.sh
```

脚本中 UCF 和 XD 使用同一套 auditor 参数。长视频按 DSANet 正式训练方式均匀压到
256 个时间 bin；hidden 和冻结 host score 使用完全相同的 bin。重复执行会复用已完成
结果，并从 `checkpoint_latest.pt` 接着未完成的 epoch 跑。

需要清空 B4 后重跑：

```bash
CLEAN=1 bash run_instructions/run_vin_vad_b4_dsanet.sh
```

## 3. 看结果

```bash
cat ../vadmy_data/vin_vad/dsanet/b4/ucf/summary.json
cat ../vadmy_data/vin_vad/dsanet/b4/xd/summary.json
tail -n 80 ../vadmy_data/vin_vad/dsanet/b4/run.log
```

通过标准：只有一个 optimizer；两项 `kappa` 都收到非零梯度；normal activation 和
normal-video q 统计都有更新；数值有限；checkpoint 同时包含 predictor、`omega`、两项
`kappa`、全部正常统计、optimizer、scheduler、配置、manifest 哈希和随机状态。

产物：

| 产物 | 相对位置 |
|---|---|
| 完整日志 | `../vadmy_data/vin_vad/dsanet/b4/run.log` |
| 每轮诊断 | `../vadmy_data/vin_vad/dsanet/b4/<dataset>/history.json` |
| 断点 | `../vadmy_data/vin_vad/dsanet/b4/<dataset>/checkpoint_latest.pt` |
| 正式模型 | `../vadmy_data/vin_vad/dsanet/b4/<dataset>/model_final.pt` |
| B4 验收 | `../vadmy_data/vin_vad/dsanet/b4/<dataset>/summary.json` |

## 一键全部运行

```bash
bash run_instructions/run_vin_vad_b4_dsanet.sh
```
