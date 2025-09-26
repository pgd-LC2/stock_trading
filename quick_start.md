# 股票预测系统快速开始指南

## 🚀 快速安装 (5分钟搞定)

### 1. 克隆仓库
```bash
git clone https://github.com/pgd-LC2/stock_trading.git
cd stock_trading
```

### 2. 安装依赖
```bash
# 方法1: 使用安装脚本 (推荐)
python setup_and_usage.py

# 方法2: 手动安装
pip install -r requirements.txt
```

### 3. 测试系统
```bash
python test_system.py
```

### 4. 开始预测
```bash
# 快速演示
python example_usage.py

# 或使用主程序
python main.py --mode demo --symbol AAPL
```

## 📊 使用示例

### 基础预测
```python
# 获取股票数据并预测
from src.data_acquisition import get_stock_data
from src.prediction import predict_stock_price

# 获取苹果股票数据
data = get_stock_data(['AAPL'], period='1y')
print(f"获取到 {len(data)} 条数据")

# 进行预测 (详细代码见 example_usage.py)
```

### 高级功能
```python
# 使用HAELT模型进行预测
from src.models.haelt_model import create_haelt_model
from src.training import train_model

# 创建和训练模型
model = create_haelt_model(input_dim=74, config={...})
trained_model = train_model(model, X_train, y_train, X_val, y_val, config)

# 生成预测
predictions = predict_stock_price(trained_model, recent_data)
```

## 🎯 主要功能

| 功能 | 命令 | 说明 |
|------|------|------|
| 数据获取 | `python main.py --mode acquire --symbol AAPL` | 获取股票历史数据 |
| 模型训练 | `python main.py --mode train --symbol AAPL` | 训练预测模型 |
| 股票预测 | `python main.py --mode predict --symbol AAPL` | 生成价格预测 |
| 模型评估 | `python main.py --mode evaluate --symbol AAPL` | 评估模型性能 |
| 完整演示 | `python main.py --mode demo --symbol AAPL` | 运行完整流程 |

## 📈 支持的股票代码

- **美股**: AAPL, TSLA, MSFT, GOOGL, AMZN, META, NVDA
- **中概股**: BABA, JD, PDD, BIDU, NIO
- **其他**: 支持Yahoo Finance的所有股票代码

## ⚙️ 配置参数

编辑 `config.yaml` 文件来调整模型参数:

```yaml
models:
  haelt:
    hidden_dim: 128      # 隐藏层维度
    num_layers: 3        # 层数
    sequence_length: 60  # 序列长度
    dropout: 0.2         # Dropout率

training:
  epochs: 100           # 训练轮数
  batch_size: 32       # 批次大小
  learning_rate: 0.001 # 学习率
```

## 🔧 故障排除

### 常见问题

1. **网络连接问题**
   ```bash
   # 设置代理 (如果需要)
   export https_proxy=http://your-proxy:port
   ```

2. **内存不足**
   ```yaml
   # 在config.yaml中减少参数
   training:
     batch_size: 16  # 减少批次大小
   ```

3. **训练速度慢**
   ```yaml
   # 减少训练轮数
   training:
     epochs: 20  # 快速测试用
   ```

### 依赖问题
```bash
# 升级pip
python -m pip install --upgrade pip

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

## 📊 输出文件说明

- `models/`: 保存训练好的模型文件
- `results/`: 预测结果和图表
- `logs/`: 运行日志和错误信息
- `data/`: 下载的股票数据缓存

## 🎉 开始使用

现在你可以开始使用这个强大的股票预测系统了！

```bash
# 运行完整演示
python example_usage.py

# 选择模式1，体验完整的预测流程
```

有问题？查看 `README.md` 获取更详细的文档。
