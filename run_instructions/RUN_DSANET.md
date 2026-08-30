# DSANet + 主方法（Universal CLS-neuron Adapter）运行说明

## 0. 环境与约定

- 远程服务器上执行，`conda activate dsanet`，所有命令都在 `vadmy_code/` 目录下运行（全部相对路径）。
- 输入只读：`../vad_data`（CLIP 特征 + 已提取好的 `[T,12,768]` hidden states，**不重复提取**）。
- 输出统一写 `../vadmy_data`。
- 步骤 1~4（数据清单 + 三个神经元专家）与其他 baseline 完全共享，跑过一次即可跳过（自带 `--resume` / 签名复用，重复执行安全）。
- 正式默认参数已显式传入，直接复制执行即可。

## 0.1 一次性准备（整个项目共享，未做过才需要）

```bash
# GT 标注目录（后续评测都要用）
mkdir -p ../vadmy_data/annotations/ucf ../vadmy_data/annotations/xd
cp baseline/DSANet/list/gt_ucf.npy ../vadmy_data/annotations/ucf/gt.npy
cp baseline/DSANet/list/gt.npy     ../vadmy_data/annotations/xd/gt.npy
```

- 权重已就位：`../vadmy_data/model/DSANet/model_ucf.pth`、`model_xd.pth`。
- 本次运行输出根目录（下文统一用 `$OUT`）：

```bash
RUN_KEY=$(git rev-parse --short HEAD)
OUT=../vadmy_data/universal_neuron_adapter/runs/$RUN_KEY
mkdir -p "$OUT"
```

## 1. 数据清单准备 + 划分审计

hidden states 清单：`../vad_data/work_<dataset>/clip_hidden_stride16_*_8gpu/manifest.csv`（已提取好，直接复用）。

```bash
# UCF
python -m universal_neuron_adapter.data --dataset ucf \
  --train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --test-csv  ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest  ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --out-dir "$OUT/ucf/data" --seed 234 --val-fraction 0.2

# XD（个别视频缺 hidden state，加 --skip-missing-hidden）
python -m universal_neuron_adapter.data --dataset xd \
  --train-csv ../vad_data/work_xd/xd_train_local.csv \
  --test-csv  ../vad_data/work_xd/xd_test_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest  ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --out-dir "$OUT/xd/data" --seed 234 --val-fraction 0.2 --skip-missing-hidden
```

输出：`$OUT/<dataset>/data/` 下的 `train_all.csv / expert_train.csv / expert_val.csv / test.csv` + `split_audit.json`（必须报告三个 overlap 全为 0）。

## 2. 方向性常态专家（数据集级共享缓存）

```bash
for dataset in ucf xd; do
  normality=../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1
  python -m universal_neuron_adapter.fit_normality_expert \
    --manifest "$OUT/$dataset/data/expert_train.csv" --out-dir "$normality" \
    --active-per-layer 32 --maximum-length 256 --resume
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$OUT/$dataset/data/test.csv" --expert-model "$normality/normality_expert.npz" --out-dir "$normality/test"
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$OUT/$dataset/data/expert_train.csv" --expert-model "$normality/normality_expert.npz" --out-dir "$normality/train"
done
```

输出：`../vadmy_data/universal_neuron_adapter/normality_expert_cache/<dataset>/top32_signed_v1/`（`normality_expert.npz` + train/test 的 `expert3_scores.csv`）。

## 3. 主稀疏专家（数据集级共享缓存）

```bash
for dataset in ucf xd; do
  primary="$OUT/$dataset/expert"
  python -m universal_neuron_adapter.train_expert \
    --train-manifest "$OUT/$dataset/data/expert_train.csv" \
    --val-manifest   "$OUT/$dataset/data/expert_val.csv" \
    --out-dir "$primary" --active-per-layer 32 --temporal-width 64 --max-epoch 20 --batch-size 8 \
    --lr 0.0003 --weight-decay 0.0001 --sparsity-weight 0.001 \
    --maximum-length 256 --num-workers 4 --seed 234 --device cuda --resume
  python -m universal_neuron_adapter.export_expert \
    --manifest "$OUT/$dataset/data/train_all.csv" --expert-model "$primary/expert_best.pth" --out-dir "$primary/train" --device cuda
  python -m universal_neuron_adapter.export_expert \
    --manifest "$OUT/$dataset/data/test.csv" --expert-model "$primary/expert_best.pth" --out-dir "$primary/test" --device cuda
done
```

输出：`$OUT/<dataset>/expert/`（`expert_best.pth` + train/test 的 `expert_scores.csv`）。

## 4. 多尺度上下文学生（数据集级共享缓存）

```bash
for dataset in ucf xd; do
  primary="$OUT/$dataset/expert"
  normality=../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1
  context=../vadmy_data/universal_neuron_adapter/context_student_cache/$dataset/top32_multiscale_seed234
  python -m universal_neuron_adapter.fit_context_student \
    --manifest "$OUT/$dataset/data/expert_train.csv" \
    --expert-manifest "$primary/train/expert_scores.csv" \
    --normality-manifest "$normality/train/expert3_scores.csv" \
    --normality-model "$normality/normality_expert.npz" \
    --out-dir "$context" --normal-samples 32 --positive-fraction 0.05 \
    --epochs 20 --seed 234 --resume
  python -m universal_neuron_adapter.export_context_student \
    --manifest "$OUT/$dataset/data/test.csv" --student-model "$context/context_student.npz" \
    --normality-model "$normality/normality_expert.npz" --out-dir "$context/test"
  python -m universal_neuron_adapter.export_context_student \
    --manifest "$OUT/$dataset/data/expert_train.csv" --student-model "$context/context_student.npz" \
    --normality-model "$normality/normality_expert.npz" --out-dir "$context/train"
done
```

输出：`../vadmy_data/universal_neuron_adapter/context_student_cache/<dataset>/top32_multiscale_seed234/`（`context_student.npz` + train/test 的 `student_scores.csv`）。

## 5. 缓存 DSANet 冻结分数（本 baseline 专属）

用官方权重跑冻结推理，得到每条视频的 binary + semantic 概率曲线（semantic 按官方 `refine_scores_hierarchical` 协议生成，训练用 `train_all.csv` 覆盖 train/val 两份 key，测试用 `test.csv`）：

```bash
for dataset in ucf xd; do
  base="$OUT/$dataset/dsanet"
  python -m universal_neuron_adapter.cache_baseline \
    --baseline dsanet --baseline-root baseline/DSANet \
    --baseline-weight "../vadmy_data/model/DSANet/model_${dataset}.pth" \
    --dataset "$dataset" --manifest "$OUT/$dataset/data/train_all.csv" --split train \
    --out-dir "$base/baseline_train" --device cuda
  python -m universal_neuron_adapter.cache_baseline \
    --baseline dsanet --baseline-root baseline/DSANet \
    --baseline-weight "../vadmy_data/model/DSANet/model_${dataset}.pth" \
    --dataset "$dataset" --manifest "$OUT/$dataset/data/test.csv" --split test \
    --out-dir "$base/baseline_test" --device cuda
done
```

输出：`$OUT/<dataset>/dsanet/baseline_{train,test}/baseline_scores.csv`（含 `baseline_score_path` 与 `semantic_score_path`）。

## 6. 训练校正头（本 baseline 专属）

```bash
for dataset in ucf xd; do
  base="$OUT/$dataset/dsanet"
  python -m universal_neuron_adapter.train_correction \
    --baseline-manifest "$base/baseline_train/baseline_scores.csv" \
    --expert-manifest "$OUT/$dataset/expert/train/expert_scores.csv" \
    --train-keys "$OUT/$dataset/data/expert_train.csv" \
    --val-keys   "$OUT/$dataset/data/expert_val.csv" \
    --baseline dsanet --dataset "$dataset" --out-dir "$base/correction" \
    --width 32 --max-epoch 15 --batch-size 32 --lr 0.0003 --weight-decay 0.0001 \
    --maximum-length 256 --num-workers 4 --seed 234 --device cuda --resume
done
```

输出：`$OUT/<dataset>/dsanet/correction/model_best.pth`（按训练集划分出的验证集选点）。

## 7. 正式评测（本 baseline 专属）

```bash
for dataset in ucf xd; do
  base="$OUT/$dataset/dsanet"
  normality=../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1
  context=../vadmy_data/universal_neuron_adapter/context_student_cache/$dataset/top32_multiscale_seed234
  python -m universal_neuron_adapter.evaluate \
    --baseline-train-manifest "$base/baseline_train/baseline_scores.csv" \
    --baseline-manifest     "$base/baseline_test/baseline_scores.csv" \
    --expert-train-manifest "$OUT/$dataset/expert/train/expert_scores.csv" \
    --expert-manifest       "$OUT/$dataset/expert/test/expert_scores.csv" \
    --expert3-manifest      "$normality/test/expert3_scores.csv" \
    --expert3-train-manifest "$normality/train/expert3_scores.csv" \
    --student-manifest      "$context/test/student_scores.csv" \
    --student-train-manifest "$context/train/student_scores.csv" \
    --correction-model "$base/correction/model_best.pth" \
    --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
    --baseline dsanet --dataset "$dataset" --out-dir "$base/evaluation" \
    --frames-per-snippet 16 --event-width 41 --event-weight 1.0 \
    --normality-smoothing-blend 0.25 --persistence-weight 1.0 \
    --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda
done
```

输出：`$OUT/<dataset>/dsanet/evaluation/metrics.json`（baseline 与 corrected 的 AUC/AP）+ `per_video.csv` + `curves/`。

## 8.（可选）官方检测 mAP（DSANet 官方 `ucf/xd_detectionMAP.py`）

```bash
for dataset in ucf xd; do
  base="$OUT/$dataset/dsanet"
  if [[ "$dataset" == "ucf" ]]; then
    segments=baseline/DSANet/list/gt_segment_ucf.npy
    labels=baseline/DSANet/list/gt_label_ucf.npy
  else
    segments=baseline/DSANet/list/gt_segment.npy
    labels=baseline/DSANet/list/gt_label.npy
  fi
  cache="$OUT/detection_map/$dataset/dsanet_cache"
  python -m universal_neuron_adapter.cache_baseline \
    --baseline dsanet --baseline-root baseline/DSANet \
    --baseline-weight "../vadmy_data/model/DSANet/model_${dataset}.pth" \
    --dataset "$dataset" --manifest "$OUT/$dataset/data/test.csv" --split test \
    --out-dir "$cache" --device cuda
  python -m universal_neuron_adapter.evaluate_detection_map \
    --dataset "$dataset" --baseline-manifest "$cache/baseline_scores.csv" \
    --evaluation-manifest "$base/evaluation/per_video.csv" \
    --segment-gt "$segments" --label-gt "$labels" --baseline-root baseline/DSANet \
    --out-path "$OUT/detection_map/$dataset/metrics.json"
done
```

输出：`$OUT/detection_map/<dataset>/metrics.json`（IoU 0.1~0.5 的 baseline/corrected mAP）。

## 9. 汇总（可选）

```bash
python -m universal_neuron_adapter.aggregate_metric --results-root ../vadmy_data/universal_neuron_adapter
```

- 一键脚本只运行 DSANet 的 UCF 与 XD 两项实验；默认实验名为 `formal_seed234`，可通过 `EXPERIMENT_NAME` 修改。
- `aggregate_metric` 现在**自动跳过还没跑的 baseline**（只汇总已存在的指标并打印 `[skip]` 提示），不会再因为缺某几个结果而崩溃。
- 同名实验会断点续跑并复用数据集级神经元专家；不同代码版本或 seed 禁止混入同一目录。

输出：`$OUT/summary.json`（相对 DSANet 论文基线 89.44% AUC / 86.95% AP 的增益）。

## 一键运行

把 1~7 步串成一条命令（可重复执行，`--resume` 保证中断续跑；会同时跑 UCF 和 XD）：

```bash
bash run_instructions/run_dsanet.sh
```

产物汇总（均在 `../vadmy_data/universal_neuron_adapter/` 下）：

| 产物 | 位置 |
| --- | --- |
| 清单与审计 | `runs/<hash>/<dataset>/data/` |
| 主专家 | `runs/<hash>/<dataset>/expert/` |
| 常态/上下文专家 | `normality_expert_cache/`、`context_student_cache/` |
| DSANet 冻结分数 | `runs/<hash>/<dataset>/dsanet/baseline_{train,test}/` |
| 校正头 | `runs/<hash>/<dataset>/dsanet/correction/` |
| 正式指标 | `runs/<hash>/<dataset>/dsanet/evaluation/metrics.json` |
