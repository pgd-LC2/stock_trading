#!/usr/bin/env python3
"""
股票预测系统主程序
使用最新的机器学习模型进行股票趋势预测

作者: Devin AI
版本: 1.0.0
"""

import argparse
import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_acquisition import get_stock_data
from src.data_preprocessing import preprocess_stock_data
from src.training import train_stock_models
from src.prediction import predict_stock_prices
from src.evaluation import evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stock_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {str(e)}")
        sys.exit(1)

def setup_directories(config: Dict[str, Any]) -> None:
    """创建必要的目录"""
    directories = [
        config['paths']['data_dir'],
        config['paths']['models_dir'],
        config['paths']['logs_dir'],
        config['paths']['results_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"目录已创建: {directory}")

def download_data(config: Dict[str, Any]) -> pd.DataFrame:
    """下载股票数据"""
    logger.info("开始下载股票数据...")
    
    symbols = config['data']['symbols']
    period = config['data']['period']
    source = config['data'].get('source', 'yahoo')
    
    tushare_token = config.get('api_keys', {}).get('tushare_token', '')
    alpha_vantage_key = config.get('api_keys', {}).get('alpha_vantage_key', '')
    
    try:
        logger.info(f"使用数据源: {source}")
        
        data = get_stock_data(
            symbols, 
            period=period, 
            source=source,
            tushare_token=tushare_token,
            alpha_vantage_key=alpha_vantage_key
        )
        
        if data.empty:
            logger.error("未能获取到股票数据")
            logger.info("尝试使用备用数据源...")
            
            backup_sources = ['tushare', 'yahoo', 'alpha_vantage']
            backup_sources = [s for s in backup_sources if s != source]
            
            for backup_source in backup_sources:
                logger.info(f"尝试备用数据源: {backup_source}")
                data = get_stock_data(
                    symbols, 
                    period=period, 
                    source=backup_source,
                    tushare_token=tushare_token,
                    alpha_vantage_key=alpha_vantage_key
                )
                if not data.empty:
                    logger.info(f"成功使用备用数据源 {backup_source} 获取数据")
                    break
            
            if data.empty:
                logger.error("所有数据源都无法获取数据")
                sys.exit(1)
        
        data_path = os.path.join(config['paths']['data_dir'], 'stock_data.csv')
        data.to_csv(data_path)
        
        logger.info(f"股票数据下载完成，共 {len(data)} 条记录")
        logger.info(f"数据已保存到: {data_path}")
        
        return data
        
    except Exception as e:
        logger.error(f"数据下载失败: {str(e)}")
        sys.exit(1)

def train_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """训练模型"""
    logger.info("开始数据预处理和模型训练...")
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = preprocess_stock_data(data, config)
        
        logger.info(f"数据预处理完成:")
        logger.info(f"  训练集: {X_train.shape}")
        logger.info(f"  验证集: {X_val.shape}")
        logger.info(f"  测试集: {X_test.shape}")
        
        results = train_stock_models(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
        
        logger.info("模型训练完成")
        
        return results
        
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        sys.exit(1)

def evaluate_models(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """评估模型性能"""
    logger.info("开始模型评估...")
    
    try:
        y_true = results['test_actuals']
        
        if 'all_predictions' in results:
            predictions_dict = results['all_predictions']
        else:
            predictions_dict = {}
            for model_name in ['haelt', 'lstm', 'transformer']:
                if model_name in results['models']:
                    predictions_dict[model_name] = results['ensemble_predictions']
            predictions_dict['ensemble'] = results['ensemble_predictions']
        
        evaluation_results = evaluate_predictions(
            y_true,
            predictions_dict,
            save_plots=True,
            plots_path=os.path.join(config['paths']['results_dir'], 'evaluation_plots.png')
        )
        
        report_path = os.path.join(config['paths']['results_dir'], 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(evaluation_results['evaluation_report'])
        
        logger.info("模型评估完成")
        logger.info(f"评估报告已保存到: {report_path}")
        
        print("\n=== 模型性能总结 ===")
        print(evaluation_results['comparison_table'])
        print(f"\n最佳模型: {evaluation_results['best_model']}")
        
    except Exception as e:
        logger.error(f"模型评估失败: {str(e)}")

def predict_future(data: pd.DataFrame, config: Dict[str, Any], days_ahead: int = 5) -> None:
    """预测未来股价"""
    logger.info(f"开始预测未来 {days_ahead} 天的股价...")
    
    try:
        prediction_results = predict_stock_prices(
            data,
            config_path="config.yaml",
            models_dir=config['paths']['models_dir'],
            days_ahead=days_ahead
        )
        
        future_predictions = prediction_results['future_predictions']
        prediction_dates = prediction_results['prediction_dates']
        
        print(f"\n=== 未来 {days_ahead} 天股价预测 ===")
        for i, (date, price) in enumerate(zip(prediction_dates, future_predictions)):
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        predictions_path = os.path.join(config['paths']['results_dir'], 'future_predictions.csv')
        predictions_df = pd.DataFrame({
            'date': prediction_dates,
            'predicted_price': future_predictions
        })
        predictions_df.to_csv(predictions_path, index=False)
        
        logger.info(f"预测结果已保存到: {predictions_path}")
        
    except Exception as e:
        logger.error(f"未来预测失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票预测系统')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'all', 'demo'], 
                       default='all', help='运行模式')
    parser.add_argument('--symbols', nargs='+', help='股票代码列表')
    parser.add_argument('--days', type=int, default=5, help='预测天数')
    parser.add_argument('--data-file', help='使用现有数据文件')
    
    args = parser.parse_args()
    
    print("=== 股票预测系统 ===")
    print("使用最新的机器学习模型进行股票趋势预测")
    print("支持的模型: HAELT, LSTM, Transformer, Ensemble")
    print()
    
    config = load_config(args.config)
    setup_directories(config)
    
    if args.symbols:
        config['data']['symbols'] = args.symbols
    
    if args.mode == 'demo':
        print("🚀 运行演示模式 - 完整股票预测流程")
        args.mode = 'all'  # demo模式等同于all模式
    
    if args.data_file and os.path.exists(args.data_file):
        logger.info(f"使用现有数据文件: {args.data_file}")
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    else:
        data = download_data(config)
    
    if args.mode in ['train', 'all']:
        results = train_models(data, config)
        
        if args.mode in ['evaluate', 'all']:
            evaluate_models(results, config)
    
    if args.mode in ['predict', 'all']:
        predict_future(data, config, args.days)
    
    print("\n=== 程序执行完成 ===")
    print("请查看 results/ 目录获取详细结果")

if __name__ == "__main__":
    main()
