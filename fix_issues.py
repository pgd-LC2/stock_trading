#!/usr/bin/env python3
"""
快速修复脚本 - 解决常见问题
Quick Fix Script - Resolve Common Issues
"""

import os
import sys
import time
import argparse

def create_directories():
    """创建必要的目录"""
    print("📁 创建必要目录...")
    
    directories = ['logs', 'data', 'models', 'results']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ 创建目录: {directory}/")
        except Exception as e:
            print(f"  ❌ 创建目录失败 {directory}: {e}")
    
    print("✅ 目录创建完成")

def fix_main_py():
    """修复main.py中的demo模式问题"""
    print("🔧 修复main.py...")
    
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "'demo'" not in content:
            content = content.replace(
                "choices=['train', 'predict', 'evaluate', 'all']",
                "choices=['train', 'predict', 'evaluate', 'all', 'demo']"
            )
            
            demo_code = """
    if args.mode == 'demo':
        print("🚀 运行演示模式 - 完整股票预测流程")
        args.mode = 'all'  # demo模式等同于all模式
"""
            
            insert_pos = content.find("if args.mode in ['train', 'all']:")
            if insert_pos != -1:
                content = content[:insert_pos] + demo_code + "\n    " + content[insert_pos:]
        
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("  ✅ main.py修复完成，现在支持--mode demo")
        
    except Exception as e:
        print(f"  ❌ 修复main.py失败: {e}")

def test_data_connection():
    """测试数据连接并提供备用方案"""
    print("🌐 测试数据连接...")
    
    try:
        import yfinance as yf
        
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info:
            print("  ✅ Yahoo Finance连接正常")
            return True
        else:
            print("  ⚠️  Yahoo Finance可能有问题")
            return False
            
    except Exception as e:
        print(f"  ❌ 数据连接测试失败: {e}")
        return False

def create_sample_data():
    """创建示例数据用于测试"""
    print("📊 创建示例数据...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        np.random.seed(42)
        base_price = 150
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        sample_data = pd.DataFrame({
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'symbol': 'AAPL'
        }, index=dates)
        
        os.makedirs('data', exist_ok=True)
        sample_data.to_csv('data/sample_AAPL.csv')
        
        print("  ✅ 示例数据创建完成: data/sample_AAPL.csv")
        print("  💡 可以使用: python main.py --mode demo --data-file data/sample_AAPL.csv")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 示例数据创建失败: {e}")
        return False

def show_usage_tips():
    """显示使用提示"""
    tips = """
🎯 使用提示:

1️⃣  如果遇到网络问题，使用示例数据:
   python main.py --mode demo --data-file data/sample_AAPL.csv

2️⃣  如果Yahoo Finance被限流，等待5-10分钟后重试:
   python main.py --mode all --symbol AAPL

3️⃣  使用其他股票代码避免限流:
   python main.py --mode demo --symbol MSFT

4️⃣  分步骤运行避免长时间等待:
   python main.py --mode train --symbol AAPL
   python main.py --mode predict --symbol AAPL

5️⃣  检查系统状态:
   python test_system.py

🔧 如果还有问题:
   - 检查网络连接
   - 确保Python版本 >= 3.8
   - 重新安装依赖: pip install -r requirements.txt
"""
    print(tips)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='快速修复脚本')
    parser.add_argument('--all', action='store_true', help='执行所有修复')
    parser.add_argument('--dirs', action='store_true', help='只创建目录')
    parser.add_argument('--fix-main', action='store_true', help='只修复main.py')
    parser.add_argument('--sample-data', action='store_true', help='只创建示例数据')
    
    args = parser.parse_args()
    
    print("🔧 股票预测系统快速修复工具")
    print("=" * 50)
    
    if args.all or not any([args.dirs, args.fix_main, args.sample_data]):
        create_directories()
        print()
        fix_main_py()
        print()
        
        if not test_data_connection():
            print()
            create_sample_data()
        
        print()
        show_usage_tips()
        
    else:
        if args.dirs:
            create_directories()
        if args.fix_main:
            fix_main_py()
        if args.sample_data:
            create_sample_data()
    
    print("\n✅ 修复完成！现在可以运行:")
    print("   python main.py --mode demo --symbol AAPL")

if __name__ == "__main__":
    main()
