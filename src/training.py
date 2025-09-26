import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
from typing import Dict, Tuple, Any
from .models.ensemble import ModelTrainingPipeline
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def create_data_loaders(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader

class TrainingManager:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['results_dir'], exist_ok=True)
    
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        
        input_dim = X_train.shape[2]
        logger.info(f"Input dimension: {input_dim}")
        
        data_loader = StockDataLoader(batch_size=self.config['models']['haelt']['batch_size'])
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        pipeline = ModelTrainingPipeline(self.config)
        pipeline.create_models(input_dim)
        
        training_results = pipeline.train_all_models(train_loader, val_loader)
        
        validation_scores = {
            name: results['best_val_loss'] 
            for name, results in training_results.items()
        }
        
        ensemble = pipeline.create_ensemble(validation_scores)
        
        ensemble_predictions, test_actuals = ensemble.predict(test_loader)
        
        results = {
            'training_results': training_results,
            'validation_scores': validation_scores,
            'ensemble_predictions': ensemble_predictions,
            'test_actuals': test_actuals,
            'models': pipeline.models,
            'ensemble': ensemble
        }
        
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        results_path = os.path.join(self.config['paths']['results_dir'], 'training_results.npz')
        
        np.savez(
            results_path,
            ensemble_predictions=results['ensemble_predictions'],
            test_actuals=results['test_actuals'],
            validation_scores=results['validation_scores']
        )
        
        for model_name, model in results['models'].items():
            model_path = os.path.join(self.config['paths']['models_dir'], f'{model_name}_final.pth')
            torch.save(model.state_dict(), model_path)
        
        if 'preprocessor' in results:
            import pickle
            scaler_path = os.path.join(self.config['paths']['models_dir'], 'scalers.pkl')
            scalers_to_save = results['preprocessor'].scalers.copy()
            scalers_to_save['feature_columns'] = results['preprocessor'].feature_columns
            with open(scaler_path, 'wb') as f:
                pickle.dump(scalers_to_save, f)
            logger.info(f"Scalers saved to {scaler_path}")
        else:
            logger.warning("No preprocessor found in results, scalers not saved")
        
        logger.info(f"Results saved to {results_path}")

def train_stock_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor=None,
    config_path: str = "config.yaml"
) -> Dict[str, Any]:
    
    trainer = TrainingManager(config_path)
    results = trainer.train_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    if preprocessor is not None:
        results['preprocessor'] = preprocessor
        trainer.save_results(results)
    
    return results
