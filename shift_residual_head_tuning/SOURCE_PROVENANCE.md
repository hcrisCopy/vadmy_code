# 代码依据

本目录没有修改 `baseline/` 或 `rely/`。

| 组件 | 参考实现 | 保留的关键细节 |
|---|---|---|
| top/bottom正负样本 | `../vad_code/experiments/intravideo_paired_shift/select_neurons_intravideo_paired.py` | 同一异常视频内等量top/bottom 10%、正常z-score、每层Top-64 |
| 768→512残差 | `../vad_code/experiments/clip512_injection/models.py` | LayerNorm、3层MLP、零初始化末层、sigmoid门控、加到原CLIP前 |
| DSANet训练配置 | `baseline/DSANet/src/ucf_option.py`、`xd_option.py` | epoch、batch、lr、UCF按1280样本验证、作者loss |
| DeSC训练配置 | `baseline/DeSC/src/ucf_option.py`、`xd_option.py` | 两流输出、epoch、batch、lr、weight decay、作者loss |
| LaGoVAD训练配置 | 作者checkpoint旁的 `../vadmy_data/model/LaGoVAD/config.yaml` | epoch=40、batch=128、lr=5e-5、weight decay=0、20步warmup后cosine |
| 三baseline接口 | `neuron_responsibility/baselines.py` | checkpoint严格加载、作者forward与loss、DeSC概率集成 |
| 官方评测 | 三个baseline发布的测试代码和检测mAP工具 | UCF AUC、XD AP、16帧重复、类别检测mAP |

跨项目的 `../vad_code` 仅用于设计时核对，不会被本目录代码导入或在服务器运行时引用。
