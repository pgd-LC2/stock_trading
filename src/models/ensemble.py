import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from .haelt_model import HAELT, HAELTTrainer
from .lstm_model import LSTMModel, LSTMTrainer
from .transformer_model import TransformerModel, TransformerTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleModel:
    def __init__(self, models: Dict[str, nn.Module], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models.keys()}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for model in self.models.values():
            model.to(self.device)
            model.eval()
    
    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        all_predictions = {}
        actuals = None
        
        for name, model in self.models.items():
            model.eval()
            predictions = []
            batch_actuals = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()
                    
                    outputs = model(batch_x)
                    
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    batch_actuals.extend(batch_y.cpu().numpy())
            
            all_predictions[name] = np.array(predictions)
            if actuals is None:
                actuals = np.array(batch_actuals)
        
        ensemble_predictions = np.zeros_like(list(all_predictions.values())[0])
        
        for name, predictions in all_predictions.items():
            ensemble_predictions += self.weights[name] * predictions
        
        logger.info(f"Ensemble prediction completed with {len(self.models)} models")
        return ensemble_predictions, actuals
    
    def update_weights(self, validation_scores: Dict[str, float]):
        total_score = sum(validation_scores.values())
        if total_score > 0:
            self.weights = {name: score / total_score for name, score in validation_scores.items()}
            logger.info(f"Updated ensemble weights: {self.weights}")

class ModelTrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.trainers = {}
        
    def create_models(self, input_dim: int):
        self.models['haelt'] = HAELT(
            input_dim=input_dim,
            hidden_dim=self.config['models']['haelt']['hidden_dim'],
            num_layers=self.config['models']['haelt']['num_layers'],
            dropout=self.config['models']['haelt']['dropout']
        )
        
        self.models['lstm'] = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.config['models']['lstm']['hidden_dim'],
            num_layers=self.config['models']['lstm']['num_layers'],
            dropout=self.config['models']['lstm']['dropout']
        )
        
        self.models['transformer'] = TransformerModel(
            input_dim=input_dim,
            d_model=self.config['models']['transformer']['d_model'],
            nhead=self.config['models']['transformer']['nhead'],
            num_layers=self.config['models']['transformer']['num_layers'],
            dropout=self.config['models']['transformer']['dropout']
        )
        
        self.trainers['haelt'] = HAELTTrainer(self.models['haelt'], self.device)
        self.trainers['lstm'] = LSTMTrainer(self.models['lstm'], self.device)
        self.trainers['transformer'] = TransformerTrainer(self.models['transformer'], self.device)
        
        for name, trainer in self.trainers.items():
            model_config = self.config['models'][name]
            trainer.setup_training(
                learning_rate=model_config['learning_rate']
            )
        
        logger.info(f"Created {len(self.models)} models for ensemble training")
    
    def train_model(self, model_name: str, train_loader, val_loader, epochs: int) -> Dict[str, float]:
        trainer = self.trainers[model_name]
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        train_losses = []
        val_losses = []
        
        logger.info(f"Training {model_name} model...")
        
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(trainer.model.state_dict(), f'models/best_{model_name}_model.pth')
            else:
                patience_counter += 1
            
            if hasattr(trainer, 'scheduler'):
                trainer.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered for {model_name} at epoch {epoch}")
                break
        
        trainer.model.load_state_dict(torch.load(f'models/best_{model_name}_model.pth'))
        
        return {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def train_all_models(self, train_loader, val_loader) -> Dict[str, Dict[str, float]]:
        results = {}
        
        for model_name in self.models.keys():
            epochs = self.config['models'][model_name]['epochs']
            results[model_name] = self.train_model(model_name, train_loader, val_loader, epochs)
        
        return results
    
    def create_ensemble(self, validation_scores: Dict[str, float] = None) -> EnsembleModel:
        if validation_scores:
            weights = {name: 1.0 / score if score > 0 else 0.0 for name, score in validation_scores.items()}
            total_weight = sum(weights.values())
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            weights = None
        
        ensemble = EnsembleModel(self.models, weights)
        logger.info("Created ensemble model")
        return ensemble

def create_ensemble_pipeline(config: Dict[str, Any]) -> ModelTrainingPipeline:
    return ModelTrainingPipeline(config)
