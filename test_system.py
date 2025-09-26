#!/usr/bin/env python3
"""
测试股票预测系统的基本功能
"""

import sys
import os
sys.path.append('src')

def test_data_acquisition():
    print("测试数据获取...")
    try:
        from src.data_acquisition import get_stock_data
        data = get_stock_data(['AAPL'], period='1mo')
        print(f"✓ 数据获取成功，形状: {data.shape}")
        return True
    except Exception as e:
        print(f"✗ 数据获取失败: {str(e)}")
        return False

def test_data_preprocessing():
    print("测试数据预处理...")
    try:
        from src.data_acquisition import get_stock_data
        from src.data_preprocessing import StockDataPreprocessor
        
        data = get_stock_data(['AAPL'], period='1mo')
        preprocessor = StockDataPreprocessor()
        processed_data = preprocessor.prepare_features(data)
        print(f"✓ 数据预处理成功，特征数: {len(preprocessor.feature_columns)}")
        return True
    except Exception as e:
        print(f"✗ 数据预处理失败: {str(e)}")
        return False

def test_model_creation():
    print("测试模型创建...")
    try:
        from src.models.haelt_model import create_haelt_model
        from src.models.lstm_model import create_lstm_model
        from src.models.transformer_model import create_transformer_model
        
        config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'd_model': 64,
            'nhead': 4
        }
        
        haelt_model = create_haelt_model(10, config)
        lstm_model = create_lstm_model(10, config)
        transformer_model = create_transformer_model(10, config)
        
        print("✓ 所有模型创建成功")
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {str(e)}")
        return False

def main():
    print("=== 股票预测系统测试 ===\n")
    
    tests = [
        test_data_acquisition,
        test_data_preprocessing,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！系统可以正常运行。")
        return True
    else:
        print("✗ 部分测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    main()
