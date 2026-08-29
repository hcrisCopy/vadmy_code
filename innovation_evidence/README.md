# 三个创新点的轻量证明

这组代码只读取现有模型、CLS hidden-state衍生分数和正式标注，不重新提特征、不训练 baseline。运行：

```bash
conda activate dsanet
bash innovation_evidence/run_all.sh
```

输出在 `../vadmy_data/innovation_evidence/<git提交号>/`，每张图都有单独的 PNG、PDF 和源数据 CSV。

怎么看：

- `innovation1/*_primary_fixed_budget`：蓝柱高于等预算控制，说明主稀疏神经元不是随机维度。
- `innovation1/*_directional_neurons`：红蓝色块说明模型同时利用“高于正常”和“低于正常”的方向性偏移。
- `innovation1/*_context_scales`：三个柱来自真实学生模型系数，说明同一组方向性神经元被放进多时间尺度，而不是另选第三套神经元。
- `innovation2/*_spectral_ablation`：比较均匀融合与谱可靠性融合；`*_spectral_weights` 显示权重会随视频变化。
- `innovation3/*_asymmetric_residual`：在同一冻结 baseline 上，比较只奖励一致异常与额外抑制冲突，隔离不对称残差的作用。

注意：这些是机制诊断图，不替代六组正式主实验，也不用于重新选择超参数。若某个对照没有提高，应如实保留 CSV，并把它报告为负结果。
