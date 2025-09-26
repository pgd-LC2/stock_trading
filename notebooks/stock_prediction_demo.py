"""
股票预测系统演示脚本

本脚本演示如何使用最新的机器学习模型进行股票趋势预测。

支持的模型:
- HAELT: 混合注意力集成学习Transformer
- LSTM: 长短期记忆网络
- Transformer: 纯Transformer架构
- Ensemble: 集成模型
"""

import sys
import os
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings
warnings.filterwarnings('ignore')

from src.data_acquisition import get_stock_data
from src.data_preprocessing import preprocess_stock_data
from src.training import train_stock_models
from src.prediction import predict_stock_prices
from src.evaluation import evaluate_predictions

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=== 股票预测系统演示 ===")
    print("使用最新的机器学习模型进行股票趋势预测")
    print()
    
    print("1. 加载配置...")
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print("配置信息:")
    print(f"股票代码: {config['data']['symbols']}")
    print(f"时间周期: {config['data']['period']}")
    print(f"特征数量: {len(config['data']['features'])}")
    print()
    
    print("2. 获取股票数据...")
    symbols = ['AAPL', 'TSLA', 'GOOGL']
    data = get_stock_data(symbols, period='2y')

    print(f"数据形状: {data.shape}")
    print(f"日期范围: {data.index.min()} 到 {data.index.max()}")
    print("数据预览:")
    print(data.head())
    print()
    
    print("3. 创建数据可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        axes[0, 0].plot(symbol_data.index, symbol_data['close'], label=symbol, alpha=0.8)
    axes[0, 0].set_title('股价走势')
    axes[0, 0].set_ylabel('收盘价 ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        axes[0, 1].plot(symbol_data.index, symbol_data['volume'], label=symbol, alpha=0.8)
    axes[0, 1].set_title('成交量')
    axes[0, 1].set_ylabel('成交量')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        axes[1, 0].hist(symbol_data['close'], bins=50, alpha=0.6, label=symbol)
    axes[1, 0].set_title('价格分布')
    axes[1, 0].set_xlabel('收盘价 ($)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        returns = symbol_data['close'].pct_change().dropna()
        axes[1, 1].hist(returns, bins=50, alpha=0.6, label=symbol)
    axes[1, 1].set_title('收益率分布')
    axes[1, 1].set_xlabel('日收益率')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/data_visualization.png', dpi=300, bbox_inches='tight')
    print("数据可视化已保存到 results/data_visualization.png")
    print()
    
    print("4. 开始数据预处理...")
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_stock_data(data, config)

    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    print()
    
    print("5. 开始模型训练...")
    print("这可能需要几分钟时间，请耐心等待...")

    results = train_stock_models(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n训练完成！")
    print("验证集损失:")
    for model_name, score in results['validation_scores'].items():
        print(f"  {model_name}: {score:.6f}")
    print()
    
    print("6. 模型性能评估...")
    y_true = results['test_actuals']
    predictions_dict = {
        'ensemble': results['ensemble_predictions']
    }

    evaluation_results = evaluate_predictions(
        y_true,
        predictions_dict,
        save_plots=True,
        plots_path='../results/evaluation_plots.png'
    )

    print("模型性能评估:")
    print(evaluation_results['comparison_table'])
    print()
    
    print("详细评估报告:")
    print(evaluation_results['evaluation_report'])
    print()
    
    print("7. 预测未来5天的股价...")
    try:
        prediction_results = predict_stock_prices(
            data,
            config_path='../config.yaml',
            models_dir='../models',
            days_ahead=5
        )
        
        future_predictions = prediction_results['future_predictions']
        prediction_dates = prediction_results['prediction_dates']
        
        print("未来5天股价预测:")
        for date, price in zip(prediction_dates, future_predictions):
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
        print()
            
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        print("请确保模型已经训练完成")
        print()
    
    print("8. 创建预测结果可视化...")
    plt.figure(figsize=(12, 8))

    recent_data = data.tail(100)
    plt.plot(recent_data.index, recent_data['close'], label='实际价格', color='blue', alpha=0.8)

    test_dates = recent_data.index[-len(y_true):]
    plt.plot(test_dates, y_true, label='测试集实际', color='green', alpha=0.8)
    plt.plot(test_dates, results['ensemble_predictions'], label='集成模型预测', color='red', alpha=0.8)

    plt.title('股价预测结果')
    plt.xlabel('日期')
    plt.ylabel('价格 ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../results/prediction_results.png', dpi=300, bbox_inches='tight')
    print("预测结果可视化已保存到 results/prediction_results.png")
    print()
    
    print("=== 演示完成 ===")
    print("总结:")
    print("1. 数据获取: 从Yahoo Finance获取实时股票数据")
    print("2. 特征工程: 计算技术指标和价格特征")
    print("3. 模型训练: 使用HAELT、LSTM、Transformer等先进模型")
    print("4. 集成学习: 结合多个模型的预测结果")
    print("5. 性能评估: 使用多种指标评估模型准确性")
    print("6. 未来预测: 预测未来几天的股价走势")
    print()
    print("关键优势:")
    print("- 使用最新的HAELT混合架构")
    print("- 集成多个模型提高准确性")
    print("- 全面的评估指标")
    print("- 易于使用的接口")
    print()
    print("注意事项:")
    print("- 股票预测存在固有风险")
    print("- 模型性能会随市场条件变化")
    print("- 建议结合其他分析方法使用")

if __name__ == "__main__":
    main()
