# 代码依据

| 部分 | 依据 |
|---|---|
| baseline专属冻结打分 | `shift_residual_head_tuning.score_baseline` |
| 同视频top/bottom与Shift-Global768 | `shift_residual_head_tuning.select_shift_neurons` |
| hidden/CLIP时间对齐 | `shift_residual_head_tuning.build_aligned_features` |
| 单残差结构 | `shift_residual_head_tuning.method.ShiftResidualInjector` |
| 全baseline冻结与小checkpoint | `shift_u_dual_injection`的冻结控制流程 |
| DeSC Sensitivity loss | DeSC论文公式：`L_MIL + L_Align` |
| DeSC UCF推理 | `baseline/DeSC/src/ucf_test_tta.py`的256-snippet滑窗与重叠平均 |
| DeSC XD推理 | `baseline/DeSC/src/xd_test_tta.py`的整段缩放与semantic异常分数 |
| loss、scheduler、选模指标 | 三个作者baseline的论文及发布配置；DeSC Sensitivity学习率为论文给出的`1e-3` |

本目录没有修改`baseline/`、`rely/`或既有实验目录。新增内容是baseline身份/权重可验证的产物契约，以及单残差全冻结训练与评估入口。

LaGoVAD发布代码同时使用`models.*`和`src.*`导入。`neuron_responsibility/baselines.py`只修正适配器的模块搜索路径，不修改作者源码。作者提供的`best.ckpt`是PreVAD初始化权重，适配器按同目录`config.yaml`构造LaGoVAD后加载任务参数，并只允许CLIP文本backbone参数由Hugging Face原始权重补齐。Lightning和TorchMetrics沿用作者环境版本；Transformers固定为兼容服务器Torch 2.5读取OpenAI CLIP权重的4.44.2。

LaGoVAD被外层adapter迁移到GPU时，Lightning内部的`device`属性不会同步。适配器在首次前向时对LaGoVAD模块自身执行同设备迁移，使作者tokenizer和模型参数使用同一设备。

DeSC官方仓库未发布训练脚本，因此没有复原或猜测Consistency的GMP训练。当前方法只训练论文定义明确的Sensitivity目标；Consistency严格加载作者权重并只参与最终概率集成。`neuron_responsibility/desc_inference.py`是对作者两个测试脚本的独立、最小改写，没有从`baseline/`运行时导入评估代码。
