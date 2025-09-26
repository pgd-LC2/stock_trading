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
            
            df['volume_sma'] = ta.volume.VolumeSMAIndicator(close=df['close'], volume=df['volume']).volume_sma()
            
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
        
        feature_data = data[self.feature_columns].values
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
            self.feature_columns = [col for col in numeric_columns if col not in ['symbol']]
        
        logger.info(f"Selected {len(self.feature_columns)} features for training")
        return df
    
    def scale_features(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, test_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, ...]:
        if self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type {self.scaler_type}, using MinMaxScaler")
            scaler = MinMaxScaler()
        
        train_scaled = train_data.copy()
        train_scaled[self.feature_columns] = scaler.fit_transform(train_data[self.feature_columns])
        
        self.scalers['features'] = scaler
        
        results = [train_scaled]
        
        if val_data is not None:
            val_scaled = val_data.copy()
            val_scaled[self.feature_columns] = scaler.transform(val_data[self.feature_columns])
            results.append(val_scaled)
        
        if test_data is not None:
            test_scaled = test_data.copy()
            test_scaled[self.feature_columns] = scaler.transform(test_data[self.feature_columns])
            results.append(test_scaled)
        
        logger.info("Feature scaling completed")
        return tuple(results)
    
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

def preprocess_stock_data(data: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test
