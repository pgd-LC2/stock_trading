#!/usr/bin/env python3
"""
股票预测系统完整安装和使用指南
Complete Setup and Usage Guide for Stock Prediction System

这个脚本包含了完整的安装步骤和使用示例
This script contains complete installation steps and usage examples
"""

import os
import sys
import subprocess
import platform

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n执行: {description}")
    print(f"命令: {command}")
    print("-" * 40)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 成功")
            if result.stdout:
                print(result.stdout)
        else:
            print("✗ 失败")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

def check_python():
    """检查Python环境"""
    print_section("检查Python环境")
    
    python_version = sys.version
    print(f"Python版本: {python_version}")
    
    if sys.version_info < (3, 8):
        print("⚠️  警告: 建议使用Python 3.8或更高版本")
    else:
        print("✓ Python版本符合要求")
    
    return True

def install_dependencies():
    """安装依赖包"""
    print_section("安装依赖包")
    
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.5.0", 
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "yfinance>=0.2.0",
        "alpha-vantage>=2.3.0",
        "ta>=0.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
        "scipy>=1.8.0"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"安装 {dep}")

def create_example_script():
    """创建示例使用脚本"""
    print_section("创建示例脚本")
    
    example_code = '''#!/usr/bin/env python3
"""
股票预测系统使用示例
Stock Prediction System Usage Example
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

def main():
    """主函数 - 完整的股票预测流程"""
    print("🚀 股票预测系统启动")
    print("=" * 50)
    
    print("\\n📊 步骤1: 获取股票数据")
    try:
        from src.data_acquisition import get_stock_data
        
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
        print(f"正在获取股票数据: {symbols}")
        
        data = get_stock_data(symbols, period='2y')  # 获取2年数据
        print(f"✓ 数据获取成功，形状: {data.shape}")
        print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
        
    except Exception as e:
        print(f"✗ 数据获取失败: {e}")
        return
    
    print("\\n🔧 步骤2: 数据预处理")
    try:
        from src.data_preprocessing import StockDataPreprocessor
        
        preprocessor = StockDataPreprocessor()
        processed_data = preprocessor.prepare_features(data)
        print(f"✓ 预处理完成，特征数: {len(preprocessor.feature_columns)}")
        
        X, y = preprocessor.create_sequences(processed_data, sequence_length=60)
        print(f"✓ 序列创建完成，形状: X={X.shape}, y={y.shape}")
        
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return
    
    print("\\n🤖 步骤3: 模型训练")
    try:
        from src.models.haelt_model import create_haelt_model
        from src.models.lstm_model import create_lstm_model
        from src.training import train_model
        
        config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'd_model': 64,
            'nhead': 4,
            'epochs': 5,  # 快速演示，实际使用建议50-100
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        input_dim = X.shape[-1]
        model = create_haelt_model(input_dim, config)
        print("✓ HAELT模型创建成功")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("开始训练模型...")
        trained_model = train_model(model, X_train, y_train, X_test, y_test, config)
        print("✓ 模型训练完成")
        
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return
    
    print("\\n🔮 步骤4: 股票预测")
    try:
        from src.prediction import predict_stock_price
        
        recent_data = X_test[-10:]  # 最近10个样本
        predictions = predict_stock_price(trained_model, recent_data)
        
        print(f"✓ 预测完成，预测了 {len(predictions)} 个数据点")
        print(f"预测价格范围: {predictions.min():.2f} - {predictions.max():.2f}")
        
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        return
    
    print("\\n📈 步骤5: 模型评估")
    try:
        from src.evaluation import evaluate_predictions
        
        actual_prices = y_test[-len(predictions):]
        metrics = evaluate_predictions(actual_prices, predictions)
        
        print("评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return
    
    print("\\n📊 步骤6: 结果可视化")
    try:
        plt.figure(figsize=(12, 6))
        
        dates = range(len(actual_prices))
        plt.plot(dates, actual_prices, label='实际价格', color='blue', linewidth=2)
        plt.plot(dates, predictions, label='预测价格', color='red', linewidth=2, linestyle='--')
        
        plt.title('股票价格预测结果', fontsize=16, fontweight='bold')
        plt.xlabel('时间点')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('stock_prediction_result.png', dpi=300, bbox_inches='tight')
        print("✓ 结果图表已保存为 'stock_prediction_result.png'")
        plt.show()
        
    except Exception as e:
        print(f"⚠️  可视化失败: {e}")
    
    print("\\n🎉 股票预测流程完成！")
    print("=" * 50)

def quick_prediction_demo():
    """快速预测演示"""
    print("\\n🚀 快速预测演示")
    print("=" * 30)
    
    try:
        from src.data_acquisition import get_stock_data
        
        symbol = 'AAPL'  # 可以修改为其他股票代码
        print(f"获取 {symbol} 的数据...")
        
        data = get_stock_data([symbol], period='6mo')
        if not data.empty:
            latest_price = data['close'].iloc[-1]
            price_change = data['close'].pct_change().iloc[-1]
            
            print(f"✓ {symbol} 最新价格: ${latest_price:.2f}")
            print(f"✓ 今日涨跌幅: {price_change*100:.2f}%")
            
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            trend = "上涨" if latest_price > sma_20 else "下跌"
            print(f"✓ 20日均线: ${sma_20:.2f}")
            print(f"✓ 短期趋势: {trend}")
        else:
            print("✗ 无法获取数据")
            
    except Exception as e:
        print(f"✗ 演示失败: {e}")

if __name__ == "__main__":
    print("股票预测系统使用示例")
    print("请选择运行模式:")
    print("1. 完整流程演示 (包含训练和预测)")
    print("2. 快速预测演示 (仅数据获取和简单分析)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_prediction_demo()
    else:
        print("无效选择，运行快速演示...")
        quick_prediction_demo()
'''
    
    with open('example_usage.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("✓ 示例脚本已创建: example_usage.py")

def main():
    """主安装流程"""
    print("🚀 股票预测系统安装向导")
    print("Stock Prediction System Setup Wizard")
    
    if not check_python():
        return
    
    print(f"\n系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python路径: {sys.executable}")
    
    response = input("\n是否继续安装依赖包? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("安装已取消")
        return
    
    install_dependencies()
    
    create_example_script()
    
    print_section("使用说明")
    print("""
📋 使用步骤:

1. 克隆仓库 (如果还没有):
   git clone https://github.com/pgd-LC2/stock_trading.git
   cd stock_trading

2. 运行安装脚本:
   python setup_and_usage.py

3. 测试系统:
   python test_system.py

4. 运行示例:
   python example_usage.py

5. 使用主程序:
   python main.py --mode demo --symbol AAPL

📊 主要功能:
- 获取实时股票数据
- 使用HAELT、LSTM、Transformer模型预测
- 生成预测图表和评估报告
- 支持多种股票代码 (AAPL, TSLA, MSFT等)

⚙️  配置文件:
- config.yaml: 模型参数配置
- requirements.txt: 依赖包列表

📈 输出文件:
- models/: 训练好的模型文件
- results/: 预测结果和图表
- logs/: 运行日志

🔧 故障排除:
- 如果遇到网络问题，请检查防火墙设置
- 如果模型训练慢，可以减少epochs参数
- 如果内存不足，可以减少batch_size参数
""")
    
    print("\n🎉 安装完成！现在可以开始使用股票预测系统了。")

if __name__ == "__main__":
    main()
