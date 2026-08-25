# 正式运行指令

在 `vadmy_code` 根目录运行。先完成 UCF 的 DSANet pilot；它没有超过作者权重时，不建议继续后五组实验。

## 一、UCF：生成一次，三个 baseline 复用

### 1. 探测并选择两层

```bash
python neuron_responsibility/discover_definition_circuits.py \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --clip-root semantic_knn_splicing/vendor/dsanet \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/layers \
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

主要产物是 `../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json`。

### 2. 提取选中完整层

```bash
python -m semantic_knn_splicing.extract_full_layers \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/full_layers \
  --skip-missing-hidden
```

输出 `full_layers/train.csv`、`test.csv` 和 float16 两层完整 CLS 文件。

### 3. 训练文本语义定位器

```bash
python -m semantic_knn_splicing.train_localizer \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/localizer \
  --max-epoch 10 \
  --batch-size 16 \
  --sequence-length 256 \
  --context-length 8 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --topk-ratio 16 \
  --smooth-weight 8e-4 \
  --sparse-weight 8e-3 \
  --val-fraction 0.10 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

输出 `localizer/localizer_best.pth`、恢复点、训练历史和参数报告。

### 4. 选择候选异常段

```bash
python -m semantic_knn_splicing.select_pseudo_segments \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --localizer-model ../vadmy_data/semantic_knn_splicing/ucf/localizer/localizer_best.pth \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/pseudo \
  --num-segments 32 \
  --topk-segments 3 \
  --device cuda
```

输出 `pseudo/pseudo_segments.csv` 和逐视频分数缓存。

### 5. 建立正常 KNN 并合成训练特征

```bash
python -m semantic_knn_splicing.build_knn_cache \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --pseudo-csv ../vadmy_data/semantic_knn_splicing/ucf/pseudo/pseudo_segments.csv \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/knn \
  --neighbors 20

python -m semantic_knn_splicing.build_synthetic_features \
  --pseudo-csv ../vadmy_data/semantic_knn_splicing/ucf/pseudo/pseudo_segments.csv \
  --retrieval-cache ../vadmy_data/semantic_knn_splicing/ucf/knn/retrieval_cache.json \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/synthetic \
  --copies-per-segment 2 \
  --max-num-clips 5 \
  --max-normal-length 96 \
  --retrieval-probability 0.5 \
  --seed 234
```

三个 baseline 共用 `synthetic/synthetic_train.csv`。

## 二、UCF：DSANet pilot

```bash
python -m semantic_knn_splicing.train_baseline \
  --baseline dsanet \
  --baseline-root semantic_knn_splicing/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --synthetic-list ../vadmy_data/semantic_knn_splicing/ucf/synthetic/synthetic_train.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/dsanet \
  --max-epoch 10 \
  --head-only-epochs 2 \
  --batch-size 64 \
  --lr 7e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --pseudo-dense-weight 1.0 \
  --pseudo-mil-weight 1.0 \
  --pseudo-topk-ratio 4 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda

python -m semantic_knn_splicing.evaluate_baseline \
  --baseline dsanet \
  --baseline-root semantic_knn_splicing/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/semantic_knn_splicing/ucf/dsanet/model_best.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/dsanet/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```

训练开始前会先评测作者权重；若所有新检查点更差，`model_best.pth` 保留作者初始化。

## 三、UCF：DSANet pilot 通过后再运行

### DeSC

```bash
python -m semantic_knn_splicing.train_baseline \
  --baseline desc \
  --baseline-root semantic_knn_splicing/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --synthetic-list ../vadmy_data/semantic_knn_splicing/ucf/synthetic/synthetic_train.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/desc \
  --max-epoch 10 \
  --head-only-epochs 2 \
  --batch-size 64 \
  --lr 5e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 1e-5 \
  --pseudo-dense-weight 1.0 \
  --pseudo-mil-weight 1.0 \
  --pseudo-topk-ratio 4 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### LaGoVAD

```bash
python -m semantic_knn_splicing.train_baseline \
  --baseline lagovad \
  --baseline-root semantic_knn_splicing/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset ucf \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --synthetic-list ../vadmy_data/semantic_knn_splicing/ucf/synthetic/synthetic_train.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/lagovad \
  --max-epoch 20 \
  --head-only-epochs 2 \
  --batch-size 64 \
  --lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0.01 \
  --pseudo-dense-weight 1.0 \
  --pseudo-mil-weight 1.0 \
  --pseudo-topk-ratio 4 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

DeSC 和 LaGoVAD 训练过程本身也会按 UCF AUC 保存最佳模型。对应的完整独立复评命令见本文最后一节，无需手动替换参数。

## 四、XD：生成一次，三个 baseline 复用

```bash
python neuron_responsibility/discover_definition_circuits.py \
  --dataset xd \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --clip-root semantic_knn_splicing/vendor/dsanet \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/layers \
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

python -m semantic_knn_splicing.extract_full_layers \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/full_layers \
  --skip-missing-hidden

python -m semantic_knn_splicing.train_localizer \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/localizer \
  --max-epoch 10 \
  --batch-size 16 \
  --sequence-length 256 \
  --context-length 8 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --topk-ratio 16 \
  --smooth-weight 8e-4 \
  --sparse-weight 8e-3 \
  --val-fraction 0.10 \
  --num-workers 4 \
  --seed 234 \
  --device cuda

python -m semantic_knn_splicing.select_pseudo_segments \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --localizer-model ../vadmy_data/semantic_knn_splicing/xd/localizer/localizer_best.pth \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/pseudo \
  --num-segments 32 \
  --topk-segments 3 \
  --device cuda

python -m semantic_knn_splicing.build_knn_cache \
  --dataset xd \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --pseudo-csv ../vadmy_data/semantic_knn_splicing/xd/pseudo/pseudo_segments.csv \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/knn \
  --neighbors 20

python -m semantic_knn_splicing.build_synthetic_features \
  --pseudo-csv ../vadmy_data/semantic_knn_splicing/xd/pseudo/pseudo_segments.csv \
  --retrieval-cache ../vadmy_data/semantic_knn_splicing/xd/knn/retrieval_cache.json \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/synthetic \
  --copies-per-segment 2 \
  --max-num-clips 5 \
  --max-normal-length 96 \
  --retrieval-probability 0.5 \
  --seed 234
```

缺失 hidden states 的 4 个 XD 训练视频会记录在 `full_layers/train_skipped.csv`，不会使流程中断。

## 五、XD：三个 baseline 完整训练命令

### DSANet

```bash
python -m semantic_knn_splicing.train_baseline \
  --baseline dsanet \
  --baseline-root semantic_knn_splicing/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --dataset xd \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --synthetic-list ../vadmy_data/semantic_knn_splicing/xd/synthetic/synthetic_train.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/dsanet \
  --max-epoch 10 \
  --head-only-epochs 2 \
  --batch-size 64 \
  --lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --pseudo-dense-weight 1.0 \
  --pseudo-mil-weight 1.0 \
  --pseudo-topk-ratio 4 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### DeSC

```bash
python -m semantic_knn_splicing.train_baseline \
  --baseline desc \
  --baseline-root semantic_knn_splicing/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --dataset xd \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --synthetic-list ../vadmy_data/semantic_knn_splicing/xd/synthetic/synthetic_train.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/desc \
  --max-epoch 10 \
  --head-only-epochs 2 \
  --batch-size 64 \
  --lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 1e-3 \
  --pseudo-dense-weight 1.0 \
  --pseudo-mil-weight 1.0 \
  --pseudo-topk-ratio 4 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### LaGoVAD

```bash
python -m semantic_knn_splicing.train_baseline \
  --baseline lagovad \
  --baseline-root semantic_knn_splicing/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset xd \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --synthetic-list ../vadmy_data/semantic_knn_splicing/xd/synthetic/synthetic_train.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/lagovad \
  --max-epoch 20 \
  --head-only-epochs 2 \
  --batch-size 64 \
  --lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0.01 \
  --pseudo-dense-weight 1.0 \
  --pseudo-mil-weight 1.0 \
  --pseudo-topk-ratio 4 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

XD 每个训练过程按 frame AP 保存 `model_best.pth`，并输出 `checkpoint_last.pth`、`history.jsonl` 和 `parameter_report.json`。

## 六、其余五组独立复评命令

训练过程已经按论文指标选模；下面只重新读取 `model_best.pth`，逐视频缓存分数并生成最终报告。

### UCF / DeSC

```bash
python -m semantic_knn_splicing.evaluate_baseline \
  --baseline desc \
  --baseline-root semantic_knn_splicing/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/semantic_knn_splicing/ucf/desc/model_best.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/desc/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```

### UCF / LaGoVAD

```bash
python -m semantic_knn_splicing.evaluate_baseline \
  --baseline lagovad \
  --baseline-root semantic_knn_splicing/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/semantic_knn_splicing/ucf/lagovad/model_best.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/ucf/lagovad/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```

### XD / DSANet

```bash
python -m semantic_knn_splicing.evaluate_baseline \
  --baseline dsanet \
  --baseline-root semantic_knn_splicing/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --model-path ../vadmy_data/semantic_knn_splicing/xd/dsanet/model_best.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/dsanet/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```

### XD / DeSC

```bash
python -m semantic_knn_splicing.evaluate_baseline \
  --baseline desc \
  --baseline-root semantic_knn_splicing/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --model-path ../vadmy_data/semantic_knn_splicing/xd/desc/model_best.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/desc/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```

### XD / LaGoVAD

```bash
python -m semantic_knn_splicing.evaluate_baseline \
  --baseline lagovad \
  --baseline-root semantic_knn_splicing/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --model-path ../vadmy_data/semantic_knn_splicing/xd/lagovad/model_best.pth \
  --out-dir ../vadmy_data/semantic_knn_splicing/xd/lagovad/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```
