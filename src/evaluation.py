import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictionEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2
        }
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        if len(y_true) < 2 or len(y_pred) < 2:
            return {'directional_accuracy': 0.0, 'hit_rate': 0.0}
        
        true_directions = np.sign(np.diff(y_true))
        pred_directions = np.sign(np.diff(y_pred))
        
        directional_accuracy = np.mean(true_directions == pred_directions)
        
        true_changes = np.diff(y_true) / y_true[:-1]
        pred_changes = np.diff(y_pred) / y_pred[:-1]
        
        hit_rate = np.mean(np.sign(true_changes) == np.sign(pred_changes))
        
        return {
            'directional_accuracy': directional_accuracy,
            'hit_rate': hit_rate
        }
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, float]:
        if len(y_true) < 2 or len(y_pred) < 2:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}
        
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        strategy_returns = []
        for i in range(len(pred_returns)):
            if pred_returns[i] > 0:
                strategy_returns.append(true_returns[i])
            elif pred_returns[i] < 0:
                strategy_returns.append(-true_returns[i])
            else:
                strategy_returns.append(0)
        
        strategy_returns = np.array(strategy_returns)
        
        if np.std(strategy_returns) == 0:
            sharpe_ratio = 0.0
        else:
            excess_returns = strategy_returns - risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'total_return': total_return,
            'volatility': np.std(strategy_returns) * np.sqrt(252)
        }
    
    def calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        
        prediction_interval_width = 2 * z_score * residual_std
        
        coverage = np.mean(np.abs(residuals) <= z_score * residual_std)
        
        return {
            'prediction_interval_width': prediction_interval_width,
            'coverage_probability': coverage,
            'residual_std': residual_std
        }
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        directional_metrics = self.calculate_directional_accuracy(y_true, y_pred)
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred)
        interval_metrics = self.calculate_prediction_intervals(y_true, y_pred)
        
        all_metrics = {
            **basic_metrics,
            **directional_metrics,
            **trading_metrics,
            **interval_metrics
        }
        
        self.metrics[model_name] = all_metrics
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"RMSE: {basic_metrics['rmse']:.4f}")
        logger.info(f"Directional Accuracy: {directional_metrics['directional_accuracy']:.4f}")
        logger.info(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
        
        return all_metrics
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        comparison_df = pd.DataFrame(results).T
        
        comparison_df = comparison_df.round(4)
        
        ranking_metrics = ['rmse', 'directional_accuracy', 'sharpe_ratio', 'hit_rate']
        
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                if metric in ['rmse', 'max_drawdown']:
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=True)
                else:
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        rank_columns = [col for col in comparison_df.columns if col.endswith('_rank')]
        if rank_columns:
            comparison_df['overall_rank'] = comparison_df[rank_columns].mean(axis=1)
        
        return comparison_df.sort_values('overall_rank') if 'overall_rank' in comparison_df.columns else comparison_df
    
    def create_evaluation_plots(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray], save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(y_true, label='Actual', alpha=0.7)
        for model_name, y_pred in predictions_dict.items():
            axes[0, 0].plot(y_pred, label=f'{model_name} Predicted', alpha=0.7)
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        for model_name, y_pred in predictions_dict.items():
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, label=model_name)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        for model_name, y_pred in predictions_dict.items():
            axes[1, 0].scatter(y_true, y_pred, alpha=0.6, label=model_name)
        
        min_val = min(y_true.min(), min(pred.min() for pred in predictions_dict.values()))
        max_val = max(y_true.max(), max(pred.max() for pred in predictions_dict.values()))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        axes[1, 0].set_title('Actual vs Predicted Scatter')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        metrics_data = []
        for model_name, y_pred in predictions_dict.items():
            metrics = self.calculate_basic_metrics(y_true, y_pred)
            metrics_data.append([model_name, metrics['rmse'], metrics['mae'], metrics['mape']])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'RMSE', 'MAE', 'MAPE'])
        
        x_pos = np.arange(len(metrics_df))
        width = 0.25
        
        axes[1, 1].bar(x_pos - width, metrics_df['RMSE'], width, label='RMSE', alpha=0.7)
        axes[1, 1].bar(x_pos, metrics_df['MAE'], width, label='MAE', alpha=0.7)
        axes[1, 1].bar(x_pos + width, metrics_df['MAPE'], width, label='MAPE', alpha=0.7)
        
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Error Metrics')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> str:
        report = "=== 股票预测模型评估报告 ===\n\n"
        
        all_results = {}
        for model_name, y_pred in predictions_dict.items():
            all_results[model_name] = self.evaluate_model_performance(y_true, y_pred, model_name)
        
        comparison_df = self.compare_models(all_results)
        
        report += "模型性能对比:\n"
        report += comparison_df.to_string() + "\n\n"
        
        best_model = comparison_df.index[0] if 'overall_rank' in comparison_df.columns else list(all_results.keys())[0]
        report += f"最佳模型: {best_model}\n\n"
        
        report += "关键指标解释:\n"
        report += "- RMSE: 均方根误差，越小越好\n"
        report += "- MAE: 平均绝对误差，越小越好\n"
        report += "- Directional Accuracy: 方向预测准确率，越高越好\n"
        report += "- Sharpe Ratio: 夏普比率，越高越好\n"
        report += "- Max Drawdown: 最大回撤，越小越好\n\n"
        
        for model_name, metrics in all_results.items():
            report += f"{model_name} 详细指标:\n"
            for metric_name, value in metrics.items():
                if not metric_name.endswith('_rank'):
                    report += f"  {metric_name}: {value:.4f}\n"
            report += "\n"
        
        return report

def evaluate_predictions(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_plots: bool = True,
    plots_path: str = "./results/evaluation_plots.png"
) -> Dict[str, Any]:
    
    evaluator = StockPredictionEvaluator()
    
    all_results = {}
    for model_name, y_pred in predictions_dict.items():
        all_results[model_name] = evaluator.evaluate_model_performance(y_true, y_pred, model_name)
    
    comparison_df = evaluator.compare_models(all_results)
    
    if save_plots:
        evaluator.create_evaluation_plots(y_true, predictions_dict, plots_path)
    
    report = evaluator.generate_evaluation_report(y_true, predictions_dict)
    
    return {
        'individual_results': all_results,
        'comparison_table': comparison_df,
        'evaluation_report': report,
        'best_model': comparison_df.index[0] if 'overall_rank' in comparison_df.columns else list(all_results.keys())[0]
    }
