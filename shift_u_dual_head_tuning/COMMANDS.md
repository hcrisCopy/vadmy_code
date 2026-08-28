# 正式命令

从 `vadmy_code` 根目录运行。六条命令已写死数据集、权重、训练参数和输出位置。

## DSANet / UCF-Crime（优先）

```bash
conda activate dsanet
bash shift_u_dual_head_tuning/commands/run_dsanet_ucf.sh
```

输出：`../vadmy_data/shift_u_dual_head_tuning/ucf/dsanet/`。

## DSANet / XD-Violence

```bash
conda activate dsanet
bash shift_u_dual_head_tuning/commands/run_dsanet_xd.sh
```

输出：`../vadmy_data/shift_u_dual_head_tuning/xd/dsanet/`。缺失的4个XD训练hidden跳过并记录，测试hidden不允许跳过。

## DeSC / UCF-Crime

```bash
conda activate dsanet
bash shift_u_dual_head_tuning/commands/run_desc_ucf.sh
```

输出：`../vadmy_data/shift_u_dual_head_tuning/ucf/desc/`。

## DeSC / XD-Violence

```bash
conda activate dsanet
bash shift_u_dual_head_tuning/commands/run_desc_xd.sh
```

输出：`../vadmy_data/shift_u_dual_head_tuning/xd/desc/`。

## LaGoVAD / UCF-Crime

```bash
conda activate dsanet
bash shift_u_dual_head_tuning/commands/run_lagovad_ucf.sh
```

输出：`../vadmy_data/shift_u_dual_head_tuning/ucf/lagovad/`。

## LaGoVAD / XD-Violence

```bash
conda activate dsanet
bash shift_u_dual_head_tuning/commands/run_lagovad_xd.sh
```

输出：`../vadmy_data/shift_u_dual_head_tuning/xd/lagovad/`。

中断后直接重跑同一命令：准备产物自动复用，存在 `checkpoint_last.pth` 时自动续训。需要重新训练时只清理本方案对应的 `training/`、`evaluation/` 和 `diagnostics/`，不要删除共享的selection与aligned feature。
