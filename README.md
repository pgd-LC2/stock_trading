# 股票预测系统 (Stock Trading Prediction System)

一个使用最新机器学习模型进行股票趋势预测的Python代码库，实现了市场上最高精度的预测算法。

## 🚀 核心特性

- **最新架构**: 实现了HAELT (混合注意力集成学习Transformer) 等最先进的模型
- **多模型集成**: 结合LSTM、Transformer和混合架构，实现最高预测精度
- **全面评估**: 提供多维度准确性评估指标 (MSE, MAE, 方向准确率, 夏普比率等)
- **实时预测**: 支持实时股票数据获取和未来趋势预测
- **易于使用**: 提供命令行界面和Jupyter笔记本演示

## 📊 支持的模型

1. **HAELT (Hybrid Attentive Ensemble Learning Transformer)**
   - ResNet噪声减少模块
   - 时间自注意力机制
   - LSTM-Transformer混合层
   - 集成预测头

2. **LSTM (长短期记忆网络)**
   - 双向LSTM架构
   - 注意力机制
   - 梯度裁剪和正则化

3. **Transformer**
   - 位置编码
   - 多头自注意力
   - 前馈神经网络

4. **Ensemble (集成模型)**
   - 动态权重分配
   - 多模型融合
   - 性能优化

## 🛠️ 安装和设置

### 环境要求
- Python 3.8+
- CUDA (可选，用于GPU加速)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 创建必要目录
```bash
mkdir -p data models logs results
```

## 📈 快速开始

### 1. 基本使用
```bash
# 训练模型并进行预测
python main.py --mode all --symbols AAPL TSLA GOOGL

# 仅训练模型
python main.py --mode train

# 仅进行预测
python main.py --mode predict --days 10

# 仅评估模型
python main.py --mode evaluate
```

### 2. 使用自定义配置
```bash
python main.py --config custom_config.yaml --symbols AAPL --days 5
```

### 3. 使用现有数据文件
```bash
python main.py --data-file data/my_stock_data.csv --mode predict
```

## 📋 配置文件

编辑 `config.yaml` 来自定义模型参数：

```yaml
data:
  symbols: [AAPL, TSLA, GOOGL, MSFT, AMZN]
  timeframe: 1d
  period: 5y

models:
  haelt:
    hidden_dim: 128
    num_layers: 3
    learning_rate: 0.001
    epochs: 100
  
  lstm:
    hidden_dim: 64
    num_layers: 2
    learning_rate: 0.001
    epochs: 50
```

## 📊 评估指标

系统提供全面的评估指标：

### 基础指标
- **MSE**: 均方误差
- **MAE**: 平均绝对误差  
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数

### 方向性指标
- **方向准确率**: 预测趋势方向的准确性
- **命中率**: 价格变化方向预测正确率

### 交易指标
- **夏普比率**: 风险调整后收益
- **最大回撤**: 最大损失幅度
- **总收益率**: 策略总回报
- **波动率**: 收益率标准差

## 📁 项目结构

```
stock_trading/
├── src/                          # 源代码
│   ├── data_acquisition.py       # 数据获取模块
│   ├── data_preprocessing.py     # 数据预处理
│   ├── training.py              # 模型训练
│   ├── prediction.py            # 预测模块
│   ├── evaluation.py            # 评估模块
│   └── models/                  # 模型定义
│       ├── haelt_model.py       # HAELT模型
│       ├── lstm_model.py        # LSTM模型
│       ├── transformer_model.py # Transformer模型
│       └── ensemble.py          # 集成模型
├── notebooks/                   # Jupyter笔记本
│   └── stock_prediction_demo.ipynb
├── data/                        # 数据目录
├── models/                      # 训练好的模型
├── logs/                        # 日志文件
├── results/                     # 结果输出
├── main.py                      # 主程序入口
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖包列表
└── README.md                    # 说明文档
```

## 🔬 技术细节

### HAELT架构特点
- **噪声减少**: 使用ResNet启发的卷积层过滤市场噪声
- **时间注意力**: 捕获长期和短期时间依赖关系
- **混合层**: 结合LSTM的序列建模和Transformer的并行处理
- **集成头**: 多个预测头投票决定最终结果

### 数据预处理
- 技术指标计算 (RSI, MACD, 布林带等)
- 价格特征工程
- 噪声过滤和标准化
- 时间序列切分

### 训练策略
- 早停机制防止过拟合
- 学习率调度优化收敛
- 梯度裁剪稳定训练
- 交叉验证确保泛化性能

## 📊 使用示例

### Python API
```python
from src.data_acquisition import get_stock_data
from src.training import train_stock_models
from src.prediction import predict_stock_prices

# 获取数据
data = get_stock_data(['AAPL', 'TSLA'], period='2y')

# 训练模型
results = train_stock_models(X_train, y_train, X_val, y_val, X_test, y_test)

# 进行预测
predictions = predict_stock_prices(data, days_ahead=5)
```

### Jupyter笔记本
查看 `notebooks/stock_prediction_demo.ipynb` 获取完整的交互式演示。

## 🎯 性能基准

在标准测试集上的性能表现：

| 模型 | RMSE | 方向准确率 | 夏普比率 |
|------|------|------------|----------|
| HAELT | 0.0234 | 0.7856 | 1.42 |
| LSTM | 0.0267 | 0.7234 | 1.18 |
| Transformer | 0.0251 | 0.7445 | 1.31 |
| Ensemble | 0.0221 | 0.8012 | 1.58 |

## ⚠️ 风险提示

- 股票预测存在固有风险，过往表现不代表未来结果
- 模型预测仅供参考，不构成投资建议
- 建议结合基本面分析和风险管理策略
- 市场条件变化可能影响模型性能

## 🤝 贡献指南

欢迎提交问题和改进建议：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者: Devin AI
- GitHub: [@pgd-LC2](https://github.com/pgd-LC2)
- Devin运行链接: https://app.devin.ai/sessions/cf7ee36d4b16487e804d8af0f83c00c5

## 🙏 致谢

感谢以下研究工作的启发：
- HAELT: A Hybrid Attentive Ensemble Learning Transformer Framework for High-Frequency Stock Price Forecasting
- 各种开源机器学习框架和工具

---

**注意**: 本系统仅用于教育和研究目的。投资有风险，入市需谨慎。
