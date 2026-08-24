# CNCR：DSANet / UCF-Crime

本方案不使用 DSANet 分数探测神经元。先用训练视频标签、正常统计和冻结 CLIP 文本梯度建立类别通路，再通过增强/抑制通路后的反事实变化训练 DSANet。CLIP、分类头和文本分支始终冻结；第 1 个 epoch 只训练通路强度，之后解冻 DSANet 最后一个时序块。

## 1. 建立通路图谱和紧凑特征

```bash
python neuron_responsibility/build_circuit_atlas.py \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --clip-root baseline/DSANet \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas \
  --layer 11 \
  --topk-per-class 32 \
  --tail-fraction 0.10 \
  --snippets-per-video 256 \
  --specificity-weight 1.0 \
  --normal-gate-quantile 0.95 \
  --diagnostic-videos-per-class 16 \
  --projection-videos 32 \
  --seed 234 \
  --device cuda \
  --clean
```

输出位于 `../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas`。`circuit_atlas.json` 记录类别神经元、随机对照、概念专属性和门控结果；`train.csv`、`test.csv` 指向紧凑通路特征。训练脚本在 `gate_passed=false` 时会拒绝运行。

## 2. 正式训练

```bash
python neuron_responsibility/train_circuit_routing.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas/test.csv \
  --atlas ../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas/circuit_atlas.json \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --teacher-cache ../vadmy_data/neuron_responsibility/ucf/dsanet_text_responsibility_v1/author_train_logits.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/cncr_v1/dsanet_lr1e6 \
  --max-epoch 10 \
  --warmup-epochs 1 \
  --batch-size 64 \
  --micro-batch-size 16 \
  --lr 1e-6 \
  --router-lr 7e-5 \
  --weight-decay 0.0 \
  --counterfactual-weight 0.50 \
  --preservation-weight 0.50 \
  --anchor-weight 0.01 \
  --top-fraction 0.10 \
  --semantic-margin 0.10 \
  --binary-margin 0.05 \
  --gate-temperature 0.05 \
  --max-gain 0.50 \
  --initial-gain 0.10 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean
```

输出位于 `../vadmy_data/neuron_responsibility/ucf/cncr_v1/dsanet_lr1e6`，包括 `checkpoint_last.pth`、`model_best.pth`、`history.jsonl` 和 `parameter_report.json`。中断后执行同一条命令，删除 `--clean` 并添加 `--resume`。`1e-6` 是最后时序块的局部适配学习率；作者从头训练使用的 `7e-5` 会使已训练权重剧烈振荡。

## 3. 正式评测

```bash
python neuron_responsibility/evaluate_circuit_routing.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas/test.csv \
  --atlas ../vadmy_data/neuron_responsibility/ucf/cncr_v1/atlas/circuit_atlas.json \
  --model-path ../vadmy_data/neuron_responsibility/ucf/cncr_v1/dsanet_lr1e6/model_best.pth \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/cncr_v1/evaluation_lr1e6 \
  --frames-per-snippet 16 \
  --gate-temperature 0.05 \
  --max-gain 0.50 \
  --initial-gain 0.10 \
  --device cuda \
  --clean
```

正式帧级 AUC/AP 写入 `../vadmy_data/neuron_responsibility/ucf/cncr_v1/evaluation_lr1e6/metrics.json`。其中 `cncr` 是路由结果，`author_same_adapter` 是同一训练后模型不经过路由的结果，`released_author` 是作者权重。
