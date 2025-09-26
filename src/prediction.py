import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import yaml
from .models.haelt_model import HAELT
from .models.lstm_model import LSTMModel
from .models.transformer_model import TransformerModel
from .models.ensemble import EnsembleModel
from .data_preprocessing import StockDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, config_path: str = "config.yaml", models_dir: str = "./models"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.preprocessor = None
        
    def load_models(self, input_dim: int):
        model_configs = self.config['models']
        
        self.models['haelt'] = HAELT(
            input_dim=input_dim,
            hidden_dim=model_configs['haelt']['hidden_dim'],
            num_layers=model_configs['haelt']['num_layers'],
            dropout=model_configs['haelt']['dropout']
        )
        
        self.models['lstm'] = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_configs['lstm']['hidden_dim'],
            num_layers=model_configs['lstm']['num_layers'],
            dropout=model_configs['lstm']['dropout']
        )
        
        self.models['transformer'] = TransformerModel(
            input_dim=input_dim,
            d_model=model_configs['transformer']['d_model'],
            nhead=model_configs['transformer']['nhead'],
            num_layers=model_configs['transformer']['num_layers'],
            dropout=model_configs['transformer']['dropout']
        )
        
        for name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f'{name}_final.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                logger.info(f"Loaded {name} model from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        self.ensemble = EnsembleModel(self.models)
    
    def prepare_data_for_prediction(self, data: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            self.preprocessor = StockDataPreprocessor()
        
        processed_data = self.preprocessor.prepare_features(data)
        
        if len(self.preprocessor.feature_columns) == 0:
            raise ValueError("No feature columns available for prediction")
        
        sequence_length = self.config['models']['haelt']['sequence_length']
        
        if len(processed_data) < sequence_length:
            raise ValueError(f"Data length {len(processed_data)} is less than required sequence length {sequence_length}")
        
        feature_data = processed_data[self.preprocessor.feature_columns].values
        
        X = []
        for i in range(sequence_length, len(feature_data)):
            X.append(feature_data[i-sequence_length:i])
        
        return np.array(X)
    
    def predict_single_model(self, model_name: str, X: np.ndarray) -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), 32):
                batch = X[i:i+32]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                outputs = model(batch_tensor)
                output_numpy = outputs.squeeze().cpu().numpy()
                if np.isscalar(output_numpy) or len(output_numpy.shape) == 0:
                    predictions.append(output_numpy.item() if hasattr(output_numpy, 'item') else output_numpy)
                else:
                    predictions.extend(output_numpy)
        
        return np.array(predictions)
    
    def predict_ensemble(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        predictions = {}
        
        for model_name in self.models.keys():
            predictions[model_name] = self.predict_single_model(model_name, X)
        
        ensemble_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
        for model_name, pred in predictions.items():
            weight = self.ensemble.weights.get(model_name, 1.0 / len(predictions))
            ensemble_pred += weight * pred
        
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def predict_trend_direction(self, predictions: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        trend_directions = []
        
        for i in range(1, len(predictions)):
            change = (predictions[i] - predictions[i-1]) / predictions[i-1]
            
            if change > threshold:
                trend_directions.append(1)
            elif change < -threshold:
                trend_directions.append(-1)
            else:
                trend_directions.append(0)
        
        return np.array(trend_directions)
    
    def predict_future_prices(self, data: pd.DataFrame, days_ahead: int = 5) -> Dict[str, Any]:
        X = self.prepare_data_for_prediction(data)
        
        if len(X) == 0:
            raise ValueError("No valid sequences for prediction")
        
        predictions = self.predict_ensemble(X)
        
        last_sequence = X[-1:]
        future_predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            future_pred = {}
            for model_name in self.models.keys():
                model_pred = self.predict_single_model(model_name, current_sequence)
                if np.isscalar(model_pred):
                    future_pred[model_name] = model_pred
                elif len(model_pred.shape) == 0:
                    future_pred[model_name] = model_pred.item()
                else:
                    future_pred[model_name] = model_pred[0]
            
            ensemble_pred = np.mean(list(future_pred.values()))
            future_predictions.append(ensemble_pred)
            
            new_features = current_sequence[0, -1, :].copy()
            new_features[0] = ensemble_pred
            
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_features
        
        trend_directions = self.predict_trend_direction(
            np.concatenate([predictions['ensemble'][-5:], future_predictions])
        )
        
        return {
            'historical_predictions': predictions,
            'future_predictions': np.array(future_predictions),
            'trend_directions': trend_directions,
            'prediction_dates': pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=days_ahead,
                freq='D'
            )
        }

class RealTimePredictionService:
    def __init__(self, predictor: StockPredictor):
        self.predictor = predictor
        self.prediction_cache = {}
        
    def get_latest_prediction(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        cache_key = f"{symbol}_{data.index[-1].strftime('%Y%m%d')}"
        
        if cache_key in self.prediction_cache:
            logger.info(f"Using cached prediction for {symbol}")
            return self.prediction_cache[cache_key]
        
        try:
            prediction_result = self.predictor.predict_future_prices(data)
            
            self.prediction_cache[cache_key] = prediction_result
            
            if len(self.prediction_cache) > 100:
                oldest_key = min(self.prediction_cache.keys())
                del self.prediction_cache[oldest_key]
            
            logger.info(f"Generated new prediction for {symbol}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {str(e)}")
            return {}
    
    def batch_predict(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        for symbol, data in symbols_data.items():
            results[symbol] = self.get_latest_prediction(symbol, data)
        
        return results

def create_predictor(config_path: str = "config.yaml", models_dir: str = "./models") -> StockPredictor:
    return StockPredictor(config_path, models_dir)

def predict_stock_prices(
    data: pd.DataFrame,
    config_path: str = "config.yaml",
    models_dir: str = "./models",
    days_ahead: int = 5
) -> Dict[str, Any]:
    
    predictor = create_predictor(config_path, models_dir)
    
    if predictor.preprocessor is None:
        predictor.preprocessor = StockDataPreprocessor()
    
    processed_data = predictor.preprocessor.prepare_features(data)
    input_dim = len(predictor.preprocessor.feature_columns)
    
    predictor.load_models(input_dim)
    
    return predictor.predict_future_prices(data, days_ahead)
