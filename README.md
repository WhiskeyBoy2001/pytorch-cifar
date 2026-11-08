# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## 保存训练记录与对比曲线

每次运行都会在 `runs/时间戳/` 下生成独立的实验记录：

- `config.json`：本次的学习率、训练轮数、batch size、momentum、weight decay、模型名称等配置。
- `metrics.csv`：每轮一行，包含 `epoch,lr,train_loss,train_acc,test_loss,test_acc`。

每轮完成训练和测试后立即写入 CSV，因此中断训练后，已完成轮次的数据仍可使用。
`epoch` 从 **0** 开始，accuracy 的单位是 **百分比（0–100）**；loss 按样本数加权平均，
`lr` 是本轮实际使用的学习率，在调度器更新前记录。

例如，保持其他参数一致，对比两个初始学习率：

```bash
python main.py --run-name lr_0.1 --lr 0.1 --epochs 200 --batch-size 128
python main.py --run-name lr_0.01 --lr 0.01 --epochs 200 --batch-size 128
```

也可以通过 `--momentum 0.9`、`--weight-decay 0.0005` 调整 SGD 参数，
通过 `--log-dir experiments` 改变记录的根目录。`--run-name` 只是曲线标签；
即使名称相同，每次运行仍会创建新的时间戳目录。
学习率采用余弦调度，`T_max` 等于本次 `--epochs`，默认训练参数与原脚本相同。

安装绘图依赖后，一条命令即可对比 `runs/` 下所有实验：

```bash
python -m pip install matplotlib
python plot_history.py
```

默认生成 `runs/comparison.png`，包括训练 loss、测试 loss、训练 acc、测试 acc 四张子图。
也可以指定实验目录或 CSV 文件，只比较选中的记录（将下面的占位目录替换成实际路径）：

```bash
python plot_history.py runs/<第一次运行的时间戳> runs/<第二次运行的时间戳> --output runs/lr_comparison.png
python plot_history.py --log-dir experiments --output experiments/comparison.png
```

加上 `--show` 可同时打开交互窗口。日志功能只使用 Python 标准库，不依赖 matplotlib；
CSV 也能直接用 Excel 或 pandas 读取。

继续使用 `--resume` 时，会从保存的最佳 checkpoint 的**下一轮**开始，并创建新的日志目录，
只记录本次运行的轮次；`--epochs` 表示本次要额外执行的轮数。与原项目一样，
这里只恢复模型权重和最佳准确率，优化器与学习率调度器会重新初始化，
并不是完整恢复中断时的训练状态。最佳权重仍保存到共享的 `checkpoint/ckpt.pth`，
不同实验可能覆盖它；如果需要保留各实验权重，请在下次训练前另行备份。

学习时建议一次只改一个参数，先用少量 epoch 确认流程，再进行完整训练。
同一参数多运行几次有助于区分参数效果与随机初始化、数据增强带来的波动。
训练指标在 `net.train()` 模式下累计，测试指标在 `net.eval()` 和 `torch.no_grad()` 下计算。
严格比较超参数时，建议从训练集划分验证集来调参，最后再用测试集评估。

可运行小规模自动化测试（使用模拟数据，不会下载 CIFAR10，也不会修改已有 checkpoint）：

```bash
python -m unittest discover -s tests -v
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

