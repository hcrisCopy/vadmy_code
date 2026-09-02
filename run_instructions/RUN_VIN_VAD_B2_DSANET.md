# DSANet 正式流程：B2 方向违背与稀疏 field

B2 只检查方向残差、`entmax-1.5` 权重、正常 running median/MAD 和逐项复算。
它不训练检测器，也不看测试集 AUC/AP。

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
bash run_instructions/run_vin_vad_b2_dsanet.sh
```

重复执行会复用已经通过的结果。需要清空 B2 后重跑：

```bash
CLEAN=1 bash run_instructions/run_vin_vad_b2_dsanet.sh
```

## 3. 看结果

```bash
cat ../vadmy_data/vin_vad/dsanet/b2/ucf/summary.json
cat ../vadmy_data/vin_vad/dsanet/b2/xd/summary.json
```

通过标准：`direction_overlap_count=0`、`probability_sum_error<=1e-5`，并且
`recompute_activation_max_abs_error` 和 `recompute_evidence_max_abs_error` 都不超过
`1e-5`。初始 `omega` 全相等，因此初始 field 是稠密的；稀疏性只允许在 B4
校正目标联合学习时产生，B2 不提前挑神经元。

产物：

| 产物 | 相对位置 |
|---|---|
| 完整日志 | `../vadmy_data/vin_vad/dsanet/b2/run.log` |
| B2 验收 | `../vadmy_data/vin_vad/dsanet/b2/<dataset>/summary.json` |
| 可复算样例 | `../vadmy_data/vin_vad/dsanet/b2/<dataset>/recompute_sample.npz` |
| 初始 field 与正常统计 | `../vadmy_data/vin_vad/dsanet/b2/<dataset>/violation_field_initial.pt` |

## 一键全部运行

```bash
bash run_instructions/run_vin_vad_b2_dsanet.sh
```
