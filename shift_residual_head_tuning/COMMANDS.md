# 正式命令

在 `vadmy_code` 根目录运行，环境按baseline激活。脚本中的正式参数已全部显式写出，不需要修改数据集或路径。

## 优先实验：DSANet / UCF-Crime

```bash
conda activate dsanet
bash shift_residual_head_tuning/commands/run_dsanet_ucf.sh
```

产物：`../vadmy_data/shift_residual_head_tuning/ucf/dsanet/`。

## DSANet / XD-Violence

```bash
conda activate dsanet
bash shift_residual_head_tuning/commands/run_dsanet_xd.sh
```

产物：`../vadmy_data/shift_residual_head_tuning/xd/dsanet/`。已知缺失的4个XD训练hidden会记录后跳过，测试hidden不允许跳过。

## DeSC / UCF-Crime

```bash
conda activate dsanet
bash shift_residual_head_tuning/commands/run_desc_ucf.sh
```

产物：`../vadmy_data/shift_residual_head_tuning/ucf/desc/`。

## DeSC / XD-Violence

```bash
conda activate dsanet
bash shift_residual_head_tuning/commands/run_desc_xd.sh
```

产物：`../vadmy_data/shift_residual_head_tuning/xd/desc/`。

## LaGoVAD / UCF-Crime

```bash
conda activate dsanet
bash shift_residual_head_tuning/commands/run_lagovad_ucf.sh
```

产物：`../vadmy_data/shift_residual_head_tuning/ucf/lagovad/`。

## LaGoVAD / XD-Violence

```bash
conda activate dsanet
bash shift_residual_head_tuning/commands/run_lagovad_xd.sh
```

产物：`../vadmy_data/shift_residual_head_tuning/xd/lagovad/`。

## 中断与清理

直接重跑同一条脚本会复用已完成的baseline分数、神经元选择和对齐特征；检测到 `checkpoint_last.pth` 时自动续训。需要重做某阶段时，只删除该实验的对应子目录，或单独运行脚本内对应Python命令并显式加 `--clean`。不要删除共享的 `../vad_data` hidden产物。
