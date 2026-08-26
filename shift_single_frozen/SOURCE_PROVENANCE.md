# 代码依据

| 部分 | 依据 |
|---|---|
| baseline专属冻结打分 | `shift_residual_head_tuning.score_baseline` |
| 同视频top/bottom与Shift-Global768 | `shift_residual_head_tuning.select_shift_neurons` |
| hidden/CLIP时间对齐 | `shift_residual_head_tuning.build_aligned_features` |
| 单残差结构 | `shift_residual_head_tuning.method.ShiftResidualInjector` |
| 全baseline冻结与小checkpoint | `shift_u_dual_injection`的冻结控制流程 |
| loss、scheduler、选模指标 | 三个作者baseline的既有适配逻辑 |

本目录没有修改`baseline/`、`rely/`或既有实验目录。新增内容是baseline身份/权重可验证的产物契约，以及单残差全冻结训练与评估入口。

LaGoVAD发布代码同时使用`models.*`和`src.*`导入。`neuron_responsibility/baselines.py`只修正适配器的模块搜索路径，不修改作者源码。作者提供的`best.ckpt`是PreVAD初始化权重，适配器按同目录`config.yaml`构造LaGoVAD后加载任务参数，并只允许CLIP文本backbone参数由Hugging Face原始权重补齐。Lightning和TorchMetrics沿用作者环境版本；Transformers固定为兼容服务器Torch 2.5读取OpenAI CLIP权重的4.44.2。
