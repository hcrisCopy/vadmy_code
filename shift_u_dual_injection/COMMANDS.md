# 正式命令

全部命令从 `vadmy_code` 根目录运行，参数、数据集、权重和输出路径均已写死，不需要替换。

## DSANet / UCF-Crime（优先）

```bash
conda activate dsanet
bash shift_u_dual_injection/commands/run_dsanet_ucf.sh
```

输出：`../vadmy_data/shift_u_dual_injection/ucf/dsanet/`。

## DSANet / XD-Violence

```bash
conda activate dsanet
bash shift_u_dual_injection/commands/run_dsanet_xd.sh
```

输出：`../vadmy_data/shift_u_dual_injection/xd/dsanet/`。已知缺失的4个XD训练hidden会跳过并记录，测试hidden不允许跳过。

## DeSC / UCF-Crime

```bash
conda activate dsanet
bash shift_u_dual_injection/commands/run_desc_ucf.sh
```

输出：`../vadmy_data/shift_u_dual_injection/ucf/desc/`。

## DeSC / XD-Violence

```bash
conda activate dsanet
bash shift_u_dual_injection/commands/run_desc_xd.sh
```

输出：`../vadmy_data/shift_u_dual_injection/xd/desc/`。

## LaGoVAD / UCF-Crime

```bash
conda activate dsanet
bash shift_u_dual_injection/commands/run_lagovad_ucf.sh
```

输出：`../vadmy_data/shift_u_dual_injection/ucf/lagovad/`。

## LaGoVAD / XD-Violence

```bash
conda activate dsanet
bash shift_u_dual_injection/commands/run_lagovad_xd.sh
```

输出：`../vadmy_data/shift_u_dual_injection/xd/lagovad/`。

直接重跑同一条脚本会复用pseudo score、神经元选择和aligned feature；存在 `checkpoint_last.pth` 时自动续训。需要重做双注入训练时，只清理 `../vadmy_data/shift_u_dual_injection/<dataset>/<baseline>/training`，不要删除共享选择产物。
