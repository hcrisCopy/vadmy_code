# 文本语义神经元责任（NTRP）

这版不使用任何 baseline 异常分数选神经元。它用冻结 CLIP 的文本方向计算第 12 层 CLS 各维度对“异常概念－正常概念”间隔的责任，并同时保留随机神经元对照。原始 hidden 和 512 维 CLIP 特征从 `../vad_data` 只读复用；新产物全部写入 `../vadmy_data`。

先执行构建和门控。门控同时检查 CLIP 投影还原、选中神经元是否超过随机对照、定位能力、跨视频稳定性以及与 DSANet 的互补性。门控失败时训练脚本会主动拒绝运行，避免浪费 4090 时间。

```bash
conda activate dsanet

python neuron_responsibility/build_text_responsibility.py \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --clip-root baseline/DSANet \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1 \
  --layer 11 \
  --topk 64 \
  --tail-fraction 0.10 \
  --snippets-per-video 256 \
  --folds 4 \
  --projection-videos 32 \
  --seed 234 \
  --device cuda \
  --clean

python neuron_responsibility/evaluate_text_responsibility.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/test.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/selected_text_neurons.json \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/gate \
  --frames-per-snippet 16 \
  --projection-cosine-min 0.90 \
  --prior-auc-min 0.60 \
  --random-auc-margin 0.01 \
  --stability-min 0.75 \
  --correlation-max 0.95 \
  --device cuda \
  --clean
```

构建产物包括 `selected_text_neurons.json`、`train.csv`、`test.csv` 和逐 snippet 四列证据：选中神经元 prior、随机对照 prior、文本责任、正常流形偏离。门控结果在 `gate/gate_metrics.json`。

门控通过后执行训练。只解冻 DSANet 最后一个时序块及其同级图结构层；分类头、文本分支和 CLIP 全部冻结。模型仍直接使用原 DSANet 输出，推理时没有新增网络模块。

```bash
python neuron_responsibility/train_text_responsibility.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gate-metrics ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/gate/gate_metrics.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_text_responsibility_v1 \
  --max-epoch 4 \
  --batch-size 32 \
  --lr 1e-6 \
  --weight-decay 0.0 \
  --prior-weight 0.5 \
  --preservation-weight 0.5 \
  --anchor-weight 0.01 \
  --low-confidence 0.20 \
  --high-confidence 0.80 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean

python neuron_responsibility/evaluate_text_responsibility.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet_text_responsibility_v1/model_best.pth \
  --dataset ucf \
  --test-list ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/test.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/text_responsibility_v1/selected_text_neurons.json \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_text_responsibility_v1/evaluation \
  --frames-per-snippet 16 \
  --projection-cosine-min 0.90 \
  --prior-auc-min 0.60 \
  --random-auc-margin 0.01 \
  --stability-min 0.75 \
  --correlation-max 0.95 \
  --device cuda \
  --clean
```

训练中断后去掉 `--clean` 并加入 `--resume`。构建和评测去掉 `--clean` 会复用已有逐视频/逐 clip 产物。XD 的 4 个缺失训练视频会记录在 `build_report.json` 并自动跳过。
