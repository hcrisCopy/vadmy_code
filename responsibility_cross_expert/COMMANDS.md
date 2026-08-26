# 正式运行命令

从 `vadmy_code` 根目录运行。命令不需要替换参数；首次运行保留 `--clean`，中断续跑时删除 `--clean`，训练命令再加 `--resume`。XD 缺失的 4 个训练视频会沿用完整层 CSV 中的记录直接跳过。

## 1. UCF：只做一次的责任层语义专家

```bash
python -m responsibility_cross_expert.build_normal_prototype \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/shared/normal_prototype \
  --validation-fraction 0.1 \
  --seed 234 \
  --save-every 50 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_semantic_expert \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/ucf/shared/normal_prototype/normal_prototype.npz \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert \
  --sequence-length 256 \
  --bottleneck 64 \
  --max-epoch 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --category-weight 1.0 \
  --validation-fraction 0.1 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean
```

主要产物：`../vadmy_data/responsibility_cross_expert/ucf/shared/normal_prototype/normal_prototype.npz` 和 `shared/semantic_expert/semantic_expert_best.pth`。

## 2. UCF：DSANet

```bash
python -m responsibility_cross_expert.export_expert_scores \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/ucf/shared/normal_prototype/normal_prototype.npz \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/dsanet/expert_scores \
  --sequence-length 256 \
  --bottleneck 64 \
  --num-workers 4 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.build_consensus_labels \
  --dataset ucf \
  --expert-score-csv ../vadmy_data/responsibility_cross_expert/ucf/dsanet/expert_scores/expert_scores.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/dsanet/consensus \
  --grouping 6 \
  --minimum-duration 4 \
  --cumulative-threshold 22.0 \
  --flat-ratio 0.55 \
  --clean

python -m responsibility_cross_expert.visualize_diagnostics \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/dsanet/consensus/consensus_labels.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/dsanet/diagnostics \
  --examples 8 \
  --clean

python -m responsibility_cross_expert.audit_gradients \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/dsanet/consensus/consensus_labels.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/dsanet/gradient_audit \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_binary_head \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/dsanet/consensus/consensus_labels.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/dsanet/head_training \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 0.01 \
  --author-loss-weight 1.0 \
  --consensus-loss-weight 1.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.evaluate_binary_head \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/responsibility_cross_expert/ucf/dsanet/head_training/model_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/dsanet/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

## 3. UCF：DeSC

```bash
python -m responsibility_cross_expert.export_expert_scores \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/ucf/shared/normal_prototype/normal_prototype.npz \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/desc/expert_scores \
  --sequence-length 256 \
  --bottleneck 64 \
  --num-workers 4 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.build_consensus_labels \
  --dataset ucf \
  --expert-score-csv ../vadmy_data/responsibility_cross_expert/ucf/desc/expert_scores/expert_scores.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/desc/consensus \
  --grouping 6 \
  --minimum-duration 4 \
  --cumulative-threshold 22.0 \
  --flat-ratio 0.55 \
  --clean

python -m responsibility_cross_expert.visualize_diagnostics \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/desc/consensus/consensus_labels.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/desc/diagnostics \
  --examples 8 \
  --clean

python -m responsibility_cross_expert.audit_gradients \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/desc/consensus/consensus_labels.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/desc/gradient_audit \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_binary_head \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/desc/consensus/consensus_labels.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/desc/head_training \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 5e-5 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 1e-5 \
  --author-loss-weight 1.0 \
  --consensus-loss-weight 1.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.evaluate_binary_head \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/responsibility_cross_expert/ucf/desc/head_training/model_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/desc/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

## 4. UCF：LaGoVAD

```bash
python -m responsibility_cross_expert.export_expert_scores \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset ucf \
  --train-csv ../vadmy_data/semantic_knn_splicing/ucf/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/ucf/shared/normal_prototype/normal_prototype.npz \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/lagovad/expert_scores \
  --sequence-length 512 \
  --bottleneck 64 \
  --num-workers 4 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.build_consensus_labels \
  --dataset ucf \
  --expert-score-csv ../vadmy_data/responsibility_cross_expert/ucf/lagovad/expert_scores/expert_scores.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/lagovad/consensus \
  --grouping 6 \
  --minimum-duration 4 \
  --cumulative-threshold 22.0 \
  --flat-ratio 0.55 \
  --clean

python -m responsibility_cross_expert.visualize_diagnostics \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/lagovad/consensus/consensus_labels.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/ucf/layers/definition_circuits.json \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/ucf/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/lagovad/diagnostics \
  --examples 8 \
  --clean

python -m responsibility_cross_expert.audit_gradients \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/lagovad/consensus/consensus_labels.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/lagovad/gradient_audit \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_binary_head \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset ucf \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/ucf/lagovad/consensus/consensus_labels.csv \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/lagovad/head_training \
  --max-epoch 40 \
  --batch-size 128 \
  --lr 5e-5 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 0.0 \
  --author-loss-weight 1.0 \
  --consensus-loss-weight 1.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 2024 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.evaluate_binary_head \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/responsibility_cross_expert/ucf/lagovad/head_training/model_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/ucf/lagovad/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

## 5. XD：只做一次的责任层语义专家

```bash
python -m responsibility_cross_expert.build_normal_prototype \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/shared/normal_prototype \
  --validation-fraction 0.1 \
  --seed 234 \
  --save-every 50 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_semantic_expert \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/xd/shared/normal_prototype/normal_prototype.npz \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert \
  --sequence-length 256 \
  --bottleneck 64 \
  --max-epoch 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --category-weight 1.0 \
  --validation-fraction 0.1 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean
```

## 6. XD：DSANet

```bash
python -m responsibility_cross_expert.export_expert_scores \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/xd/shared/normal_prototype/normal_prototype.npz \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/dsanet/expert_scores \
  --sequence-length 256 \
  --bottleneck 64 \
  --num-workers 4 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.build_consensus_labels \
  --dataset xd \
  --expert-score-csv ../vadmy_data/responsibility_cross_expert/xd/dsanet/expert_scores/expert_scores.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/dsanet/consensus \
  --grouping 6 \
  --minimum-duration 4 \
  --cumulative-threshold 22.0 \
  --flat-ratio 0.55 \
  --clean

python -m responsibility_cross_expert.visualize_diagnostics \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/dsanet/consensus/consensus_labels.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/dsanet/diagnostics \
  --examples 8 \
  --clean

python -m responsibility_cross_expert.audit_gradients \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/dsanet/consensus/consensus_labels.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/dsanet/gradient_audit \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_binary_head \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/dsanet/consensus/consensus_labels.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/dsanet/head_training \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 0.01 \
  --author-loss-weight 1.0 \
  --consensus-loss-weight 1.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.evaluate_binary_head \
  --baseline dsanet \
  --baseline-root responsibility_cross_expert/vendor/dsanet/src \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --model-path ../vadmy_data/responsibility_cross_expert/xd/dsanet/head_training/model_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/dsanet/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

## 7. XD：DeSC

```bash
python -m responsibility_cross_expert.export_expert_scores \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/xd/shared/normal_prototype/normal_prototype.npz \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/desc/expert_scores \
  --sequence-length 256 \
  --bottleneck 64 \
  --num-workers 4 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.build_consensus_labels \
  --dataset xd \
  --expert-score-csv ../vadmy_data/responsibility_cross_expert/xd/desc/expert_scores/expert_scores.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/desc/consensus \
  --grouping 6 \
  --minimum-duration 4 \
  --cumulative-threshold 22.0 \
  --flat-ratio 0.55 \
  --clean

python -m responsibility_cross_expert.visualize_diagnostics \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/desc/consensus/consensus_labels.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/desc/diagnostics \
  --examples 8 \
  --clean

python -m responsibility_cross_expert.audit_gradients \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/desc/consensus/consensus_labels.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/desc/gradient_audit \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_binary_head \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/desc/consensus/consensus_labels.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/desc/head_training \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 1e-3 \
  --author-loss-weight 1.0 \
  --consensus-loss-weight 1.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.evaluate_binary_head \
  --baseline desc \
  --baseline-root responsibility_cross_expert/vendor/desc_src \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --model-path ../vadmy_data/responsibility_cross_expert/xd/desc/head_training/model_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/desc/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

## 8. XD：LaGoVAD

```bash
python -m responsibility_cross_expert.export_expert_scores \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset xd \
  --train-csv ../vadmy_data/semantic_knn_splicing/xd/full_layers/train.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --clip-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --normal-prototype ../vadmy_data/responsibility_cross_expert/xd/shared/normal_prototype/normal_prototype.npz \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/lagovad/expert_scores \
  --sequence-length 512 \
  --bottleneck 64 \
  --num-workers 4 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.build_consensus_labels \
  --dataset xd \
  --expert-score-csv ../vadmy_data/responsibility_cross_expert/xd/lagovad/expert_scores/expert_scores.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/lagovad/consensus \
  --grouping 6 \
  --minimum-duration 4 \
  --cumulative-threshold 22.0 \
  --flat-ratio 0.55 \
  --clean

python -m responsibility_cross_expert.visualize_diagnostics \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/lagovad/consensus/consensus_labels.csv \
  --layer-atlas ../vadmy_data/semantic_knn_splicing/xd/layers/definition_circuits.json \
  --semantic-checkpoint ../vadmy_data/responsibility_cross_expert/xd/shared/semantic_expert/semantic_expert_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/lagovad/diagnostics \
  --examples 8 \
  --clean

python -m responsibility_cross_expert.audit_gradients \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/lagovad/consensus/consensus_labels.csv \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/lagovad/gradient_audit \
  --device cuda \
  --clean

python -m responsibility_cross_expert.train_binary_head \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset xd \
  --consensus-csv ../vadmy_data/responsibility_cross_expert/xd/lagovad/consensus/consensus_labels.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/lagovad/head_training \
  --max-epoch 40 \
  --batch-size 128 \
  --lr 5e-5 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 0.0 \
  --author-loss-weight 1.0 \
  --consensus-loss-weight 1.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 2024 \
  --device cuda \
  --clean

python -m responsibility_cross_expert.evaluate_binary_head \
  --baseline lagovad \
  --baseline-root responsibility_cross_expert/vendor/lagovad_src \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --model-path ../vadmy_data/responsibility_cross_expert/xd/lagovad/head_training/model_best.pth \
  --out-dir ../vadmy_data/responsibility_cross_expert/xd/lagovad/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

## 9. 看哪些产物

先检查每个 baseline 的：

- `diagnostics/expert_agreement_matrix.png`：两专家是否互补；如果几乎全在对角线，第二专家没有带来新信息。
- `diagnostics/class_layer_evidence_heatmap.png`：不同异常类别是否由不同完整层负责；如果所有行相同，层解释不成立。
- `diagnostics/layer_gate_weights.png`：责任探测先验与目标域训练后的整层权重是否一致。
- `diagnostics/temporal_consensus_examples.png`：保留段是否连续、是否只剩极少峰值。
- `gradient_audit/gradient_path_audit.png`：必须只有最终二分类头有梯度。
- `head_training/history.jsonl` 和 `evaluation/metrics.json`：与作者初始化指标比较，最佳模型仍按 UCF AUC / XD AP 选择。

如果语义专家验证接近随机、共识覆盖几乎为零或梯度越过最终头，应停止训练，而不是继续解冻更多层。
