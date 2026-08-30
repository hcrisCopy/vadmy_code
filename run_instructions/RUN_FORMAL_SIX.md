# 六项正式实验

一次运行 LaGoVAD、DeSC、DSANet 在 UCF-Crime 和 XD-Violence 上的六项实验：

```bash
conda activate dsanet
EXPERIMENT_NAME=formal_seed234 bash run_instructions/run_formal_six.sh
```

输出目录：

```text
../vadmy_data/universal_neuron_adapter/runs/formal_seed234/
```

相同实验名用于断点续跑。若要真正从头独立重跑，请换一个易读名称，例如 `formal_seed234_repeat2`。脚本会拒绝把不同代码版本或不同 seed 的产物混进同一个实验目录。
