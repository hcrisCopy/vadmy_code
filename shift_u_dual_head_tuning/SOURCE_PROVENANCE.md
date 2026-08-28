# 代码依据

本目录没有修改 `baseline/`、`rely/`、`shift_residual_head_tuning/` 或 `shift_u_dual_injection/`。

| 组件 | 复用依据 |
|---|---|
| 正负样本、Shift-Global768选择、aligned feature | `shift_residual_head_tuning/` |
| U形共享主干与early/late hook | `shift_u_dual_injection/method.py` |
| 三baseline打分头映射 | `shift_residual_head_tuning/method.py::configure_score_head_only` |
| 作者loss、forward和checkpoint载入 | `neuron_responsibility/baselines.py`中基于三个发布baseline复制适配的接口 |
| lr、weight decay、scheduler、epoch和选模 | 三个baseline发布配置，与前两条对照保持一致 |

本目录新增的是小checkpoint中的打分头独立保存/恢复、打分头相对变化统计，以及双注入与head联合训练流程。
