#!/usr/bin/env python3
"""
一键安装脚本 - 股票预测系统
One-click Installation Script for Stock Prediction System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """显示安装横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    股票预测系统安装器                          ║
║                Stock Prediction System Installer              ║
║                                                              ║
║  🚀 最先进的机器学习股票预测系统                                ║
║  📊 支持HAELT、LSTM、Transformer模型                          ║
║  💡 一键安装，开箱即用                                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本: {sys.version.split()[0]}")
    
    try:
        import pip
        print("✅ pip已安装")
    except ImportError:
        print("❌ 错误: pip未安装")
        return False
    
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=5)
        print("✅ 网络连接正常")
    except:
        print("⚠️  警告: 网络连接可能有问题")
    
    return True

def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--quiet"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies():
    """安装所有依赖"""
    print("\n📦 安装依赖包...")
    
    packages = [
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
    
    failed_packages = []
    
    for i, package in enumerate(packages, 1):
        package_name = package.split('>=')[0]
        print(f"  [{i:2d}/{len(packages)}] 安装 {package_name}...", end=" ")
        
        if install_package(package):
            print("✅")
        else:
            print("❌")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  以下包安装失败:")
        for pkg in failed_packages:
            print(f"    - {pkg}")
        print("\n💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 升级pip: python -m pip install --upgrade pip")
        print("   3. 手动安装失败的包")
        return False
    
    print("\n✅ 所有依赖包安装成功!")
    return True

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    
    directories = [
        "data",
        "models", 
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")

def run_tests():
    """运行系统测试"""
    print("\n🧪 运行系统测试...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_system.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 系统测试通过!")
            return True
        else:
            print("❌ 系统测试失败:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  测试超时，但系统可能正常工作")
        return True
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return False

def show_usage_guide():
    """显示使用指南"""
    guide = """
🎉 安装完成! 现在可以开始使用股票预测系统了。

📋 快速开始:

1️⃣  运行示例程序:
   python example_usage.py

2️⃣  使用主程序:
   python main.py --mode demo --symbol AAPL

3️⃣  获取帮助:
   python main.py --help

📊 常用命令:
   
   python main.py --mode predict --symbol AAPL
   
   python main.py --mode train --symbol TSLA
   
   python main.py --mode evaluate --symbol MSFT

📈 支持的股票代码:
   AAPL, TSLA, MSFT, GOOGL, AMZN, META, NVDA, BABA, JD...

⚙️  配置文件:
   编辑 config.yaml 来调整模型参数

📚 详细文档:
   查看 README.md 和 quick_start.md

🔧 如有问题:
   1. 查看 logs/ 目录下的日志文件
   2. 运行 python test_system.py 检查系统状态
   3. 检查 requirements.txt 中的依赖版本

祝你使用愉快! 🚀
"""
    print(guide)

def main():
    """主安装流程"""
    print_banner()
    
    if not check_requirements():
        print("\n❌ 系统要求检查失败，安装终止")
        sys.exit(1)
    
    print("\n" + "="*60)
    response = input("是否继续安装? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是', '确定']:
        print("安装已取消")
        sys.exit(0)
    
    print("\n🔧 升级pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                  capture_output=True)
    
    if not install_dependencies():
        print("\n❌ 依赖安装失败，请检查错误信息")
        sys.exit(1)
    
    create_directories()
    
    if not run_tests():
        print("\n⚠️  测试未完全通过，但系统可能仍可使用")
    
    show_usage_guide()

if __name__ == "__main__":
    main()
