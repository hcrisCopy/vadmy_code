# DSANet 正式流程：B0 数据与 host 身份

这一阶段只确认三件事：hidden、DSANet 正式分数和 GT 对齐；校正关闭时逐点等于 host；后续实验共用同一套评估口径。这里不训练 CVA-VAD。

## 1. 登录和更新代码

```bash
ssh -p 19423 root@connect.cqa1.seetacloud.com
cd /root/autodl-tmp/vadmy_code
source /etc/network_turbo
git pull
conda activate dsanet
```

## 2. 正式运行

```bash
bash run_instructions/run_vin_vad_b0_dsanet.sh
```

脚本会先跑 B0 单元测试，再依次处理 UCF-Crime 和 XD-Violence。中断后执行同一条命令，会复用已经完成的分数和评估。

需要清空 B0 旧产物并重跑时：

```bash
CLEAN=1 bash run_instructions/run_vin_vad_b0_dsanet.sh
```

## 3. 看结果

```bash
cat ../vadmy_data/vin_vad/dsanet/b0/ucf/evaluation/metrics.json
cat ../vadmy_data/vin_vad/dsanet/b0/xd/evaluation/metrics.json
```

重点看：

- `host_identity_max_abs_error` 必须是 `0`；
- UCF 的 `pooled_auc`、XD 的 `pooled_ap` 必须复现 DSANet 正式 evaluator；
- `cross_auc` 和 `macro_within_auc` 分别描述跨视频排序和单个异常视频内的时间排序；
- `normal_fpr.normal_video_frame_fpr` 是 95% TPR 时，全正常视频中的逐帧误报率。

产物都在 `../vadmy_data/vin_vad/dsanet/b0/`：

| 产物 | 相对位置 |
|---|---|
| 完整日志 | `run.log` |
| 数据与边界审计 | `<dataset>/data/audit.json` |
| 冻结 host 分数 | `<dataset>/host/{train,test}/scores/` |
| host 对齐审计 | `<dataset>/evaluation/audit.json` |
| 正式指标 | `<dataset>/evaluation/metrics.json` |
| 每视频摘要 | `<dataset>/evaluation/per_video.csv` |
| 每视频原始曲线 | `<dataset>/evaluation/curves/<video>.npz` |

## 一键全部运行

```bash
bash run_instructions/run_vin_vad_b0_dsanet.sh
```
