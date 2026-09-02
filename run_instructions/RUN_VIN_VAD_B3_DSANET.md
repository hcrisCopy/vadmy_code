# DSANet 正式流程：B3 两轴 host auditor

B3 只检查校正器是否严格按公式工作：identity、cross 单侧下压、within 零均值、
改动上界、分支独立和 padding 隔离。它不训练模型，也不计算 AUC/AP。

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
bash run_instructions/run_vin_vad_b3_dsanet.sh
```

重复执行会复用已经通过的结果。需要清空 B3 后重跑：

```bash
CLEAN=1 bash run_instructions/run_vin_vad_b3_dsanet.sh
```

## 3. 看结果

```bash
cat ../vadmy_data/vin_vad/dsanet/b3/ucf/summary.json
cat ../vadmy_data/vin_vad/dsanet/b3/xd/summary.json
```

通过标准：identity、符号、常数方向、边界、分支独立、padding 输出和 budget 的误差
均不超过 `1e-6`，within masked mean 不超过 `1e-6`，两个 `kappa` 在零点都有非零
梯度。脚本中的 `alpha/kappa` 只是结构审计探针，不参与模型选择；B4 才开始训练。

产物：

| 产物 | 相对位置 |
|---|---|
| 完整日志 | `../vadmy_data/vin_vad/dsanet/b3/run.log` |
| B3 验收 | `../vadmy_data/vin_vad/dsanet/b3/<dataset>/summary.json` |
| 可复算校正数组 | `../vadmy_data/vin_vad/dsanet/b3/<dataset>/audit_arrays.npz` |

## 一键全部运行

```bash
bash run_instructions/run_vin_vad_b3_dsanet.sh
```
