# 代码依据

本目录没有修改 `baseline/`、`rely/` 或上一方案代码。

| 内容 | 依据 |
|---|---|
| top/bottom正负构建、每层Top-64、aligned feature | `shift_residual_head_tuning/` 中已经按旧 `vad_code` Shift-Global768迁移并审查的实现 |
| 早期零初始化残差 | `shift_residual_head_tuning/method.py`，来源记录见其 `SOURCE_PROVENANCE.md` |
| 后期共同hook | `neuron_responsibility/baselines.py` 已有的 post-temporal feature hook |
| DSANet后期位置 | `baseline/DSANet/src/model.py::encode_video` 返回值到 `classifier`、文本对齐和DNP之间 |
| DeSC后期位置 | `baseline/DeSC/src/model_modular_corrected.py`、`model_gmp.py`、`model_multigmp.py` 的 `encode_video` 返回值到各head之间 |
| LaGoVAD后期位置 | `baseline/LaGoVAD-PreVAD/src/models/LaGoVAD/lagovad.py::_temporal_encoding` 返回值到fusion/head之间 |

U形双注入模块是本项目新增部分；其设计约束是共享神经元主干、双零初始化出口、独立门控、baseline零可训练参数。
