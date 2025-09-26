#!/usr/bin/env python3
"""
股票数据刷新工具
Stock Data Refresh Tool

用于刷新本地股票数据库，获取最新的市场数据
Used to refresh local stock database with latest market data
"""

import os
import sys
import shutil
import argparse
import pandas as pd
from datetime import datetime, timedelta
import logging

sys.path.append('src')

from src.data_acquisition import StockDataAcquisition, get_stock_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_banner():
    """显示刷新工具横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    股票数据刷新工具                            ║
║                  Stock Data Refresh Tool                     ║
║                                                              ║
║  🔄 刷新本地股票数据库                                         ║
║  📊 获取最新市场数据                                           ║
║  🗑️  清除过期缓存                                             ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def clear_cache():
    """清除所有缓存数据"""
    print("🗑️  清除缓存数据...")
    
    cache_dirs = ['data/', 'models/', 'results/', 'logs/']
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                for filename in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"  ✅ 删除文件: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"  ✅ 删除目录: {file_path}")
                        
            except Exception as e:
                print(f"  ❌ 清除 {cache_dir} 失败: {e}")
        else:
            print(f"  ℹ️  目录不存在: {cache_dir}")
    
    print("✅ 缓存清除完成")

def check_data_freshness(data_file: str) -> bool:
    """检查数据是否新鲜（24小时内）"""
    if not os.path.exists(data_file):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(data_file))
    current_time = datetime.now()
    
    if current_time - file_time > timedelta(hours=24):
        return False
    
    return True

def refresh_stock_data(symbols: list, period: str = "2y", force: bool = False):
    """刷新股票数据"""
    print(f"📊 刷新股票数据: {symbols}")
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    for symbol in symbols:
        data_file = os.path.join(data_dir, f"{symbol}_data.csv")
        
        if not force and check_data_freshness(data_file):
            print(f"  ℹ️  {symbol} 数据较新，跳过刷新")
            continue
        
        print(f"  🔄 正在刷新 {symbol} 数据...")
        
        try:
            acquisition = StockDataAcquisition()
            data = acquisition.fetch_yahoo_data(symbol, period=period)
            
            if not data.empty:
                acquisition.save_data(data, data_file)
                print(f"  ✅ {symbol} 数据刷新成功，共 {len(data)} 条记录")
                
                start_date = data.index.min().strftime('%Y-%m-%d')
                end_date = data.index.max().strftime('%Y-%m-%d')
                latest_price = data['close'].iloc[-1]
                print(f"     时间范围: {start_date} 到 {end_date}")
                print(f"     最新价格: ${latest_price:.2f}")
            else:
                print(f"  ❌ {symbol} 数据获取失败")
                
        except Exception as e:
            print(f"  ❌ {symbol} 刷新失败: {e}")

def refresh_all_data(force: bool = False):
    """刷新所有常用股票数据"""
    print("📈 刷新所有常用股票数据...")
    
    popular_symbols = [
        'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
        'BABA', 'JD', 'PDD', 'BIDU', 'NIO',
        'SPY', 'QQQ', 'BTC-USD', 'ETH-USD'
    ]
    
    refresh_stock_data(popular_symbols, force=force)

def update_existing_data():
    """更新现有数据文件"""
    print("🔄 更新现有数据文件...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("  ℹ️  数据目录不存在，无需更新")
        return
    
    existing_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not existing_files:
        print("  ℹ️  未找到现有数据文件")
        return
    
    print(f"  找到 {len(existing_files)} 个数据文件")
    
    for file in existing_files:
        file_path = os.path.join(data_dir, file)
        
        try:
            existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if 'symbol' in existing_data.columns:
                symbol = existing_data['symbol'].iloc[0]
                
                print(f"  🔄 更新 {symbol} 数据...")
                
                acquisition = StockDataAcquisition()
                new_data = acquisition.fetch_yahoo_data(symbol, period="1mo")
                
                if not new_data.empty:
                    combined_data = pd.concat([existing_data, new_data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data = combined_data.sort_index()
                    
                    combined_data.to_csv(file_path)
                    print(f"    ✅ {symbol} 数据更新完成")
                else:
                    print(f"    ❌ {symbol} 新数据获取失败")
            else:
                print(f"    ⚠️  {file} 格式不正确，跳过")
                
        except Exception as e:
            print(f"    ❌ 更新 {file} 失败: {e}")

def show_data_status():
    """显示数据状态"""
    print("📊 数据状态检查...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("  ❌ 数据目录不存在")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not files:
        print("  ℹ️  未找到数据文件")
        return
    
    print(f"  📁 数据目录: {data_dir}")
    print(f"  📄 文件数量: {len(files)}")
    print()
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        
        try:
            file_size = os.path.getsize(file_path) / 1024  # KB
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            age = datetime.now() - file_time
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'Unknown'
            
            status = "🟢 新鲜" if age.total_seconds() < 86400 else "🟡 较旧"
            
            print(f"  {status} {symbol:6} | {len(data):5}条 | {file_size:6.1f}KB | {file_time.strftime('%Y-%m-%d %H:%M')}")
            
        except Exception as e:
            print(f"  ❌ {file:20} | 读取失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票数据刷新工具')
    parser.add_argument('--mode', choices=['refresh', 'clear', 'update', 'status', 'all'], 
                       default='refresh', help='操作模式')
    parser.add_argument('--symbols', nargs='+', help='指定股票代码')
    parser.add_argument('--force', action='store_true', help='强制刷新所有数据')
    parser.add_argument('--period', default='2y', help='数据时间范围')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode == 'clear':
        clear_cache()
        
    elif args.mode == 'refresh':
        if args.symbols:
            refresh_stock_data(args.symbols, period=args.period, force=args.force)
        else:
            refresh_all_data(force=args.force)
            
    elif args.mode == 'update':
        update_existing_data()
        
    elif args.mode == 'status':
        show_data_status()
        
    elif args.mode == 'all':
        print("🚀 执行完整刷新流程...")
        clear_cache()
        refresh_all_data(force=True)
        show_data_status()
    
    print("\n✅ 数据刷新完成！")
    print("💡 提示: 运行 'python refresh_data.py --mode status' 查看数据状态")

if __name__ == "__main__":
    main()
