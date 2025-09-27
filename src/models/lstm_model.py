import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        
        last_output = lstm_out[:, -1, :]
        
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

class DirectionalLoss(nn.Module):
    def __init__(self, price_weight=0.7, direction_weight=0.3):
        super(DirectionalLoss, self).__init__()
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        price_loss = self.mse_loss(predictions, targets)
        
        if len(predictions) > 1 and len(targets) > 1:
            pred_directions = torch.sign(predictions[1:] - predictions[:-1])
            true_directions = torch.sign(targets[1:] - targets[:-1])
            
            direction_loss = 1.0 - torch.mean((pred_directions == true_directions).float())
        else:
            direction_loss = torch.tensor(0.0, device=predictions.device)
        
        total_loss = self.price_weight * price_loss + self.direction_weight * direction_loss
        return total_loss

class LSTMTrainer:
    def __init__(self, model: LSTMModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.criterion = DirectionalLoss()
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            min_lr=1e-6
        )
        
    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)

def create_lstm_model(input_dim: int, config: dict) -> LSTMModel:
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    logger.info(f"Created LSTM model with {sum(p.numel() for p in model.parameters())} parameters")
    return model
