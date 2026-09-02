# DSANet 正式流程：E1 evidence 生死对照

本阶段固定 DSANet host、训练、auditor、loss、budget 和 evaluator，只比较 C0--C4。
不调参，不做 smoothing，不用测试集选 checkpoint。

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
bash run_instructions/run_vin_vad_e1_dsanet.sh
```

脚本会依次训练并测试 UCF、XD 的 C0--C4，全部使用 seed 42 和固定最后一轮模型。
中断后重复同一条命令即可续跑；已经完成的训练、视频曲线和审计会直接复用。

清空 E1 后重跑：

```bash
CLEAN=1 bash run_instructions/run_vin_vad_e1_dsanet.sh
```

## 3. 看结果

```bash
column -s, -t < ../vadmy_data/vin_vad/dsanet/e1/comparison.csv | less -S
cat ../vadmy_data/vin_vad/dsanet/e1/summary.json
cat ../vadmy_data/vin_vad/dsanet/e1/ucf/c3/context_replacement/summary.json
cat ../vadmy_data/vin_vad/dsanet/e1/xd/c3/context_replacement/summary.json
tail -n 100 ../vadmy_data/vin_vad/dsanet/e1/run.log
```

UCF 主指标是 pooled AUC，XD 主指标是 pooled AP。C3 只有同时满足下面三件事才保留
contextual-directional claim：B1 conditional NLL 在两个数据集都优于 global；C3 主指标
在两个数据集都严格优于 C0/C1/C2/C4；固定目标 raw activation 后，替换上下文能稳定
改变 `mu -> residual -> evidence -> correction -> corrected score`。

主要产物：

| 产物 | 相对位置 |
|---|---|
| 完整日志 | `../vadmy_data/vin_vad/dsanet/e1/run.log` |
| 总对比表 | `../vadmy_data/vin_vad/dsanet/e1/comparison.csv` |
| E1 结论 | `../vadmy_data/vin_vad/dsanet/e1/summary.json` |
| 每组训练 | `../vadmy_data/vin_vad/dsanet/e1/<dataset>/<C0-C4>/training/` |
| 每组指标和曲线 | `../vadmy_data/vin_vad/dsanet/e1/<dataset>/<C0-C4>/evaluation/` |
| C3 替换审计 | `../vadmy_data/vin_vad/dsanet/e1/<dataset>/c3/context_replacement/` |

## 一键全部运行

```bash
bash run_instructions/run_vin_vad_e1_dsanet.sh
```

## 4. 2026-09-02 正式结论

代码 commit：`94de5a3`。正式汇总为 `no-go`：C3 在 UCF 和 XD 的主指标上都没有严格
优于 C0/C1/C2/C4；UCF context replacement 失败，XD 通过。当前 C3 不进入 E2。

核对本次结果：

```bash
cd /root/autodl-tmp/vadmy_code
git rev-parse HEAD
column -s, -t < ../vadmy_data/vin_vad/dsanet/e1/comparison.csv | less -S
cat ../vadmy_data/vin_vad/dsanet/e1/summary.json
cat ../vadmy_data/vin_vad/dsanet/e1/ucf/c3/context_replacement/summary.json
cat ../vadmy_data/vin_vad/dsanet/e1/xd/c3/context_replacement/summary.json
```
