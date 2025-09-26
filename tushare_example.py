#!/usr/bin/env python3
"""
Tushare数据获取示例
使用Tushare API获取股票数据的示例脚本
"""

import sys
import os
sys.path.append('src')

from src.data_acquisition import get_stock_data
import pandas as pd

def main():
    print("=== Tushare数据获取示例 ===")
    
    TUSHARE_TOKEN = "3bfd74dfdcb21818ca1a765234144cf13f0f359988362c1e3bfbefc5"
    
    symbols = ['AAPL', '000001', '600036']  # 苹果、平安银行、招商银行
    
    print(f"正在获取股票数据: {symbols}")
    print("数据源: Tushare")
    
    try:
        data = get_stock_data(
            symbols=symbols,
            period='1y',
            source='tushare',
            tushare_token=TUSHARE_TOKEN
        )
        
        if not data.empty:
            print(f"\n✅ 数据获取成功!")
            print(f"数据形状: {data.shape}")
            print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
            print(f"包含股票: {data['symbol'].unique()}")
            
            print("\n📊 最新数据预览:")
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                latest = symbol_data.iloc[-1]
                print(f"{symbol}: 最新价格 {latest['close']:.2f}, 成交量 {latest['volume']:,.0f}")
            
            os.makedirs('data', exist_ok=True)
            data.to_csv('data/tushare_sample_data.csv')
            print(f"\n💾 数据已保存到: data/tushare_sample_data.csv")
            
        else:
            print("❌ 未能获取到数据")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
