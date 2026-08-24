# DANCE：DSANet / UCF-Crime

DANCE 先在 12 层 CLS hidden states 中寻找“达到 95% 类别证据所需神经元最少”的层，再选质量最高的两层。探测只使用视频标签、正常统计和冻结 CLIP 的文本梯度，不读取 DSANet 分数。训练时，回路只决定同一视频内哪些片段应排在前面；测试仍是作者 DSANet 原始前向。

代码复用本仓库 `build_circuit_atlas.py` 的正常统计、文本梯度和类别特异性回路实现，以及 DSANet 发布代码 `src/ucf_train.py` 的原损失与选模规则。CLIP 始终冻结；依次解冻分类头、最后时序/图模块、DNP 正常模式模块。

## 1. 跨层探测并构建紧凑特征

```bash
python neuron_responsibility/discover_definition_circuits.py \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --clip-root baseline/DSANet \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits \
  --layers all \
  --selected-layers 2 \
  --k-grid 8,16,32,64,128 \
  --sufficiency-ratio 0.95 \
  --tail-fraction 0.10 \
  --snippets-per-video 256 \
  --specificity-weight 1.0 \
  --seed 234 \
  --device cuda \
  --skip-missing-hidden
```

输出为 `../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json`、`train.csv`、`test.csv` 和紧凑回路特征。原 hidden states 直接从 `../vad_data` 复用，不复制到 `vadmy_data`。中断后重跑同一指令会复用已完成特征；需要清空旧结果时追加 `--clean`。

## 2. 训练

```bash
python neuron_responsibility/train_definition_evidence.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --atlas ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dance_v1/dsanet \
  --max-epoch 10 \
  --temporal-start-epoch 2 \
  --reference-start-epoch 4 \
  --batch-size 64 \
  --head-lr 7e-5 \
  --temporal-lr 7e-6 \
  --reference-lr 7e-6 \
  --weight-decay 0 \
  --top-fraction 0.10 \
  --binary-margin 0.20 \
  --semantic-margin 0.20 \
  --binary-weight 0.25 \
  --semantic-weight 0.25 \
  --normal-weight 0.10 \
  --dnp-weight 0.20 \
  --anchor-weight 0.01 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

输出为 `../vadmy_data/neuron_responsibility/ucf/dance_v1/dsanet/` 下的 `model_best.pth`、`checkpoint_last.pth`、`history.jsonl` 和参数报告。与作者一致，每训练 1280 个样本按帧级 AUC 选最优模型。训练中断后使用完全相同指令并追加 `--resume`。

## 3. 正式评测

```bash
python neuron_responsibility/evaluate_definition_evidence.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/neuron_responsibility/ucf/dance_v1/dsanet/model_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dance_v1/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

输出 `metrics.json` 和 `per_video.csv`。评测同时报告作者权重与新权重；两者都不读取回路特征，因此没有新增推理开销。
