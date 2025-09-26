import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from typing import List, Dict, Optional, Tuple
import logging
from alpha_vantage.timeseries import TimeSeries
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataAcquisition:
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        if alpha_vantage_key:
            self.ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
    
    def fetch_yahoo_data(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            data.columns = [col.lower() for col in data.columns]
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_alpha_vantage_data(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not provided")
            return pd.DataFrame()
        
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            data.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in data.columns]
            data['symbol'] = symbol
            data.index = pd.to_datetime(data.index)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Alpha Vantage")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], source: str = "yahoo", **kwargs) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            
            if source == "yahoo":
                data = self.fetch_yahoo_data(symbol, **kwargs)
            elif source == "alpha_vantage":
                data = self.fetch_alpha_vantage_data(symbol, **kwargs)
            else:
                logger.error(f"Unknown data source: {source}")
                continue
            
            if not data.empty:
                data_dict[symbol] = data
            
            time.sleep(0.1)
        
        return data_dict
    
    def combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data_dict:
            return pd.DataFrame()
        
        combined_data = pd.concat(data_dict.values(), ignore_index=False)
        combined_data = combined_data.sort_index()
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        return combined_data
    
    def save_data(self, data: pd.DataFrame, filepath: str) -> None:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {filepath}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return pd.DataFrame()

def get_stock_data(symbols: List[str], period: str = "5y", source: str = "yahoo") -> pd.DataFrame:
    acquisition = StockDataAcquisition()
    data_dict = acquisition.fetch_multiple_symbols(symbols, source=source, period=period)
    return acquisition.combine_data(data_dict)
