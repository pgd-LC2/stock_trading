import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.scalers = {}
        self.feature_columns = []
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        try:
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
            
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            bollinger = ta.volatility.BollingerBands(close=df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            df['sma_5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
            df['sma_10'] = ta.trend.SMAIndicator(close=df['close'], window=10).sma_indicator()
            df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
            
            df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
            
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            
            df['stoch_k'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
            df['stoch_d'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch_signal()
            
            df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            logger.info("Technical indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
        
        return df
    
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        df['volume_change'] = df['volume'].pct_change()
        df['volume_price_trend'] = df['volume'] * df['price_change']
        
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()
            df[f'return_{window}'] = df['close'].pct_change(periods=window)
        
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        df['trend_strength'] = df['close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0)
        df['trend_acceleration'] = df['trend_strength'] - df['trend_strength'].shift(5)
        
        df['price_position'] = (df['close'] - df['close'].rolling(window=20).min()) / (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        
        df['volume_price_correlation'] = df['volume'].rolling(window=10).corr(df['close'])
        df['volume_trend'] = df['volume'] / df['volume'].rolling(window=10).mean()
        df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
        
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['support_level'] = df['low'].rolling(window=20).min()
        df['price_vs_resistance'] = df['close'] / df['resistance_level']
        df['price_vs_support'] = df['close'] / df['support_level']
        
        df['price_direction_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
        df['price_direction_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
        df['volume_direction'] = np.where(df['volume'] > df['volume'].shift(1), 1, -1)
        
        df['sma_5_20_cross'] = np.where(df['sma_5'] > df['sma_20'], 1, -1)
        df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['ema_12_26_cross'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
        
        df['volatility_regime'] = np.where(df['volatility_20'] > df['volatility_20'].rolling(window=50).mean(), 1, -1)
        df['volume_regime'] = np.where(df['volume'] > df['volume'].rolling(window=50).mean(), 1, -1)
        
        logger.info("Price features added successfully")
        return df
    
    def resnet_noise_reduction(self, data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = data.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns:
                original = df[col].values
                smoothed = df[col].rolling(window=window, center=True).mean().fillna(df[col])
                
                residual = original - smoothed.values
                filtered_residual = np.where(np.abs(residual) > 2 * np.std(residual), 0, residual)
                
                df[f'{col}_filtered'] = smoothed.values + filtered_residual
        
        logger.info("ResNet-inspired noise reduction applied")
        return df
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 60, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < sequence_length:
            logger.warning(f"Data length {len(data)} is less than sequence length {sequence_length}")
            return np.array([]), np.array([])
        
        feature_cols = [col for col in self.feature_columns if col != target_column]
        feature_data = data[feature_cols].values
        target_data = data[target_column].values
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(feature_data[i-sequence_length:i])
            y.append(target_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def prepare_features(self, data: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        df = self.calculate_technical_indicators(data)
        df = self.add_price_features(df)
        df = self.resnet_noise_reduction(df)
        
        df = df.dropna()
        
        if feature_list:
            available_features = [col for col in feature_list if col in df.columns]
            self.feature_columns = available_features
        else:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            original_columns = [col for col in numeric_columns if not col.endswith('_filtered') and col not in ['symbol']]
            self.feature_columns = original_columns
        
        logger.info(f"Selected {len(self.feature_columns)} features for training")
        return df
    
    def scale_features(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, test_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, ...]:
        if self.scaler_type == "minmax":
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type {self.scaler_type}, using MinMaxScaler")
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        
        feature_cols = [col for col in self.feature_columns if col != 'close']
        
        train_scaled = train_data.copy()
        train_scaled[feature_cols] = feature_scaler.fit_transform(train_data[feature_cols])
        
        if 'close' in train_data.columns:
            train_scaled[['close']] = target_scaler.fit_transform(train_data[['close']])
        
        self.scalers['features'] = feature_scaler
        self.scalers['target'] = target_scaler
        
        results = [train_scaled]
        
        if val_data is not None:
            val_scaled = val_data.copy()
            val_scaled[feature_cols] = feature_scaler.transform(val_data[feature_cols])
            if 'close' in val_data.columns:
                val_scaled[['close']] = target_scaler.transform(val_data[['close']])
            results.append(val_scaled)
        
        if test_data is not None:
            test_scaled = test_data.copy()
            test_scaled[feature_cols] = feature_scaler.transform(test_data[feature_cols])
            if 'close' in test_data.columns:
                test_scaled[['close']] = target_scaler.transform(test_data[['close']])
            results.append(test_scaled)
        
        logger.info("Feature and target scaling completed")
        return tuple(results)
    
    def inverse_transform_target(self, scaled_predictions: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original price range"""
        if 'target' not in self.scalers:
            logger.warning("Target scaler not found, returning predictions as-is")
            return scaled_predictions
        
        predictions_2d = scaled_predictions.reshape(-1, 1)
        real_predictions = self.scalers['target'].inverse_transform(predictions_2d)
        return real_predictions.flatten()
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_sorted = data.sort_index()
        
        n = len(data_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data_sorted.iloc[:train_end]
        val_data = data_sorted.iloc[train_end:val_end]
        test_data = data_sorted.iloc[val_end:]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data

def preprocess_stock_data(data: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StockDataPreprocessor]:
    preprocessor = StockDataPreprocessor()
    
    processed_data = preprocessor.prepare_features(data)
    
    train_data, val_data, test_data = preprocessor.split_data(
        processed_data, 
        config['training']['train_split'], 
        config['training']['val_split']
    )
    
    train_scaled, val_scaled, test_scaled = preprocessor.scale_features(train_data, val_data, test_data)
    
    sequence_length = config['models']['haelt']['sequence_length']
    
    X_train, y_train = preprocessor.create_sequences(train_scaled, sequence_length)
    X_val, y_val = preprocessor.create_sequences(val_scaled, sequence_length)
    X_test, y_test = preprocessor.create_sequences(test_scaled, sequence_length)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
