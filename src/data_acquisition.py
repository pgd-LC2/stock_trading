import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from typing import List, Dict, Optional, Tuple
import logging
from alpha_vantage.timeseries import TimeSeries
import os
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataAcquisition:
    def __init__(self, alpha_vantage_key: Optional[str] = None, tushare_token: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.tushare_token = tushare_token
        
        if alpha_vantage_key:
            self.ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
        
        if tushare_token and TUSHARE_AVAILABLE:
            ts.set_token(tushare_token)
            self.tushare_pro = ts.pro_api()
            logger.info("Tushare API initialized successfully")
        elif tushare_token and not TUSHARE_AVAILABLE:
            logger.warning("Tushare token provided but tushare package not installed")
        
        self.tushare_pro = None if not (tushare_token and TUSHARE_AVAILABLE) else self.tushare_pro
    
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
    
    def fetch_tushare_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """使用Tushare获取股票数据"""
        if not self.tushare_pro:
            logger.warning("Tushare API not initialized")
            return pd.DataFrame()
        
        try:
            if symbol.isdigit() and len(symbol) == 6:
                if symbol.startswith('00') or symbol.startswith('30'):
                    ts_symbol = f"{symbol}.SZ"
                else:
                    ts_symbol = f"{symbol}.SH"
            else:
                ts_symbol = f"{symbol}.O" if '.' not in symbol else symbol
            
            if not end_date:
                end_date = pd.Timestamp.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y%m%d')
            
            data = self.tushare_pro.daily(ts_code=ts_symbol, start_date=start_date, end_date=end_date)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol} ({ts_symbol})")
                return pd.DataFrame()
            
            data['trade_date'] = pd.to_datetime(data['trade_date'])
            data = data.set_index('trade_date').sort_index()
            
            data = data.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            })
            
            data['symbol'] = symbol
            
            data = data[['open', 'high', 'low', 'close', 'volume', 'symbol']]
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Tushare")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Tushare data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_multiple_symbols(self, symbols: List[str], source: str = "yahoo", **kwargs) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol} from {source}")
            
            if source == "yahoo":
                data = self.fetch_yahoo_data(symbol, **kwargs)
            elif source == "alpha_vantage":
                data = self.fetch_alpha_vantage_data(symbol, **kwargs)
            elif source == "tushare":
                data = self.fetch_tushare_data(symbol, **kwargs)
            else:
                logger.error(f"Unknown data source: {source}")
                continue
            
            if not data.empty:
                data_dict[symbol] = data
            else:
                logger.warning(f"No data retrieved for {symbol} from {source}")
            
            time.sleep(0.2)  # 增加延迟避免API限制
        
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

def get_stock_data(symbols: List[str], period: str = "5y", source: str = "yahoo", 
                   tushare_token: Optional[str] = None, alpha_vantage_key: Optional[str] = None) -> pd.DataFrame:
    """
    获取股票数据的主要函数
    
    Args:
        symbols: 股票代码列表
        period: 时间周期 (yahoo finance格式: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        source: 数据源 ("yahoo", "tushare", "alpha_vantage")
        tushare_token: Tushare API token
        alpha_vantage_key: Alpha Vantage API key
    
    Returns:
        合并后的股票数据DataFrame
    """
    acquisition = StockDataAcquisition(
        alpha_vantage_key=alpha_vantage_key,
        tushare_token=tushare_token
    )
    
    if source == "tushare":
        if period == "1y":
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y%m%d')
        elif period == "2y":
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y%m%d')
        elif period == "5y":
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y%m%d')
        else:
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y%m%d')
        
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
        data_dict = acquisition.fetch_multiple_symbols(
            symbols, source=source, start_date=start_date, end_date=end_date
        )
    else:
        data_dict = acquisition.fetch_multiple_symbols(symbols, source=source, period=period)
    
    return acquisition.combine_data(data_dict)
