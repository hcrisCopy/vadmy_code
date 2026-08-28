# CACC：DSANet / UCF-Crime

CACC 使用全部 12 层 CLIP CLS hidden states 作为候选，通过 64 维低秩层注意力学习跨层概念子空间。正常统计负责结构偏离，经过 CLIP 文本初始化并可学习的正常/异常锚点负责语义验证；两者共同门控输入 DSANet 时序模块前的小残差。探测和门控不读取 DSANet 异常分数。

原始 hidden states 直接从 `../vad_data` 按 manifest 读取，不复制到 `vadmy_data`。所有新产物都写入 `../vadmy_data/neuron_responsibility/ucf/cacc_v1/`。

## 1. 准备索引、正常统计和文本锚点

```bash
python neuron_responsibility/prepare_cacc.py \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --clip-root baseline/DSANet \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared \
  --normal-stat-snippets-per-video 256 \
  --device cuda \
  --skip-missing-hidden
```

输出 `prepared/cacc_artifact.npz`、`train.csv`、`test.csv` 和 `prepare_report.json`。再次运行会复用完整产物；需要清空时追加 `--clean`。

## 2. 正式训练

```bash
python neuron_responsibility/train_cacc.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/test.csv \
  --artifact ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/cacc_artifact.npz \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --teacher-cache ../vadmy_data/neuron_responsibility/ucf/dsanet_boundary_conditioning_v1/author_train_logits.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/cacc_v1/dsanet \
  --max-epoch 10 \
  --temporal-start-epoch 2 \
  --reference-start-epoch 4 \
  --batch-size 64 \
  --concept-width 64 \
  --temporal-kernel 5 \
  --semantic-temperature 0.07 \
  --max-residual-scale 0.25 \
  --circuit-lr 5e-5 \
  --anchor-lr 1e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --reference-lr 1e-6 \
  --weight-decay 0 \
  --mil-weight 0.10 \
  --normal-weight 0.10 \
  --compact-weight 0.01 \
  --smooth-weight 0.01 \
  --layer-sparse-weight 0.001 \
  --semantic-anchor-weight 0.01 \
  --preservation-weight 0.50 \
  --baseline-anchor-weight 0.01 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --cache-videos 8 \
  --seed 234 \
  --device cuda
```

输出 `dsanet/model_best.pth`、`checkpoint_last.pth`、`history.jsonl` 和参数报告。与作者 DSANet 一致，每处理 1280 个样本按 UCF 帧级 AUC 选模；初始作者权重也参与选择，训练退化时不会覆盖作者最优模型。中断后使用完全相同的指令并追加 `--resume`，程序从已保存的 epoch/batch 继续。

## 3. 正式评测和残差消融

```bash
python neuron_responsibility/evaluate_cacc.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/neuron_responsibility/ucf/cacc_v1/dsanet/model_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/cacc_v1/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

输出 `evaluation/metrics.json`、`per_video.csv` 和可复用的逐视频分数。指标同时包含作者 DSANet、完整 CACC、关闭 CACC 残差后的微调 baseline、独立偏离/语义门控，以及 12 层平均权重。
