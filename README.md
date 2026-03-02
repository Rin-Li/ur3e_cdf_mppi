# 3-DOF UR3e CDF Project

**Configuration-Dependent Distance Field (CDF)** 训练与测试，针对UR3e机器人前3个关节。

## 项目结构

```
three_freedom/
├── data/                       # 训练数据
│   └── cdf_3dof_data_*.pt     # 生成的数据集
├── checkpoints/                # 模型检查点
│   ├── model_epoch*.pt        # 定期保存
│   └── best_model_epoch*.pt   # 最佳模型
├── logs/                       # 训练日志
│   └── training_*.log         # CSV格式日志
├── utils/                      # 工具函数（预留）
├── data_generator_3dof.py      # 数据生成器
├── train_cdf_3dof.py           # 训练脚本
├── test_cdf_3dof.py            # 测试脚本
└── README.md                   # 本文档
```

## 系统配置

- **机器人**: UR3e
- **自由度**: 3-DOF (前3个关节可动，后3个关节固定为0°)
- **输入维度**: 6D (x, y, z, q1, q2, q3)
- **输出维度**: 1D (distance field)
- **神经网络**: MLPRegression [256, 256, 128, 128, 128]

## 工作空间定义

- **X轴**: [0.0, 0.8] 米
- **Y轴**: [-0.6, 0.6] 米
- **Z轴**: [0.0, 0.7] 米

相比2-DOF版本扩大了范围以适应3个关节的更大可达空间。

## 快速开始

### 1. 数据生成

生成3-DOF训练数据（零级集点对）：

```bash
# 小规模测试（grid=10，约1000个点，~30个配置/点）
python data_generator_3dof.py --grid 10 --batch-size 100 --max-configs 50

# 中等规模（grid=20，约8000个点，建议500k+配置）
python data_generator_3dof.py --grid 20 --batch-size 200 --max-configs 100

# 大规模（grid=30，约27000个点，用于最终训练）
python data_generator_3dof.py --grid 30 --batch-size 500 --max-configs 150
```

**参数说明**:
- `--grid`: 工作空间网格划分（grid³个点）
- `--batch-size`: 优化批次大小
- `--max-configs`: 每个任务空间点的最大配置数

**生成时间估计**:
- Grid=10: ~10分钟
- Grid=20: ~1-2小时
- Grid=30: ~10-20小时

输出文件: `data/cdf_3dof_data_{grid}.pt`

### 2. 模型训练

使用生成的数据训练CDF神经网络：

```bash
# 快速测试训练（1000轮）
python train_cdf_3dof.py --data data/cdf_3dof_data_10.pt --epochs 1000 --batch-size 50

# 标准训练（20k轮，对应2-DOF最佳epoch）
python train_cdf_3dof.py --data data/cdf_3dof_data_20.pt --epochs 20000 --batch-size 100 --lr 0.01

# 长期训练（50k轮）
python train_cdf_3dof.py --data data/cdf_3dof_data_30.pt --epochs 50000 --batch-size 200
```

**参数说明**:
- `--data`: 训练数据路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率（默认0.01）
- `--weight-decay`: L2正则化系数（默认1e-5）

**训练时间估计**:
- 1000 epochs: ~5分钟
- 20000 epochs: ~2小时
- 50000 epochs: ~5小时

模型保存在 `checkpoints/` 目录：
- `model_epoch{N}.pt`: 每100轮保存
- `best_model_epoch{N}.pt`: 最佳损失模型

### 3. 模型测试

使用PyBullet验证CDF预测准确性：

```bash
# 测试100个随机配置
python test_cdf_3dof.py --model checkpoints/best_model_epoch18900.pt --num-samples 100

# 测试1000个配置（更全面）
python test_cdf_3dof.py --model checkpoints/best_model_epoch18900.pt --num-samples 1000
```

**测试指标**:
- **平均误差**: CDF预测距离 vs PyBullet真实距离
- **碰撞判断错误率**: 符号不一致的比例（d<0 vs d>0）
- **最大误差**: 最差情况分析

**期望结果** (参考2-DOF性能):
- 平均误差 < 0.05m
- 碰撞判断错误率 < 10%

## 与2-DOF版本的差异

| 项目 | 2-DOF | 3-DOF |
|------|-------|-------|
| 活动关节 | q1, q2 | q1, q2, q3 |
| 固定关节 | q3-q6 = 0° | q4-q6 = 0° |
| 输入维度 | 5D (x,y,z,q1,q2) | 6D (x,y,z,q1,q2,q3) |
| 工作空间 | X[0,0.537], Y[-0.5,0.5], Z[0.1,0.6] | X[0,0.8], Y[-0.6,0.6], Z[0,0.7] |
| 数据量 | 4552点, 130k配置 | 8000点, 500k+配置 |
| 训练时间 | ~2小时 (20k epochs) | ~5小时 (50k epochs) |

## 数据格式

训练数据 `cdf_3dof_data_{grid}.pt` 是一个Python字典：

```python
{
    0: {
        'x': torch.Tensor([x, y, z]),        # 任务空间点
        'q': torch.Tensor([[q1, q2, q3],    # N个配置
                           [q1, q2, q3],
                           ...])
    },
    1: { ... },
    ...
}
```

每个键是任务空间点的索引，值包含：
- `x`: [3] 任务空间坐标
- `q`: [N, 3] 该点对应的N个零级集配置

## 已知问题

基于2-DOF版本的经验教训：

1. **CDF近似误差**: 神经网络可能无法完美拟合复杂几何，导致：
   - 距离预测偏差
   - 碰撞/无碰撞判断错误

2. **需要集成PyBullet碰撞检测**: 在规划算法（如MPPI）中，应该：
   - 使用CDF梯度进行快速优化
   - 使用PyBullet进行最终碰撞验证

3. **训练数据质量至关重要**:
   - Grid太粗糙 → 工作空间覆盖不足
   - max_configs太少 → 配置空间采样不足
   - 建议Grid≥20, max_configs≥100

## 下一步计划

本项目专注于**数据生成和模型训练**，不包含规划算法。

如需使用训练好的3-DOF CDF进行路径规划：
- 可参考2-DOF版本的 `rrt_planner_2dof.py`
- 或集成到MPPI等基于优化的规划器
- **务必结合PyBullet碰撞检测以确保安全性**

## 参考

- 2-DOF项目: `/home/kklab-ur-robot/ur_sdf/cdf/`
- 原始CDF实现: `/home/kklab-ur-robot/ur_sdf/cdf/nn_cdf.py`
- UR3e PyBullet接口: `/home/kklab-ur-robot/ur_sdf/sdf_ur/robot_pybullet.py`

## 许可

[根据原项目许可]
