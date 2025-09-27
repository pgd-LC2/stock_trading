import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNetNoiseReduction(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ResNetNoiseReduction, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(input_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = x.transpose(1, 2)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        
        out = out.transpose(1, 2)
        
        out += residual
        return F.relu(out)

class TemporalSelfAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(TemporalSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        residual = x
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        
        output = self.layer_norm(context + residual)
        return output

class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super(HybridLSTMTransformer, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm_output_dim = hidden_dim * 2
        
        self.attention = TemporalSelfAttention(
            input_dim=self.lstm_output_dim,
            num_heads=8,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(self.lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        attention_out = self.attention(lstm_out)
        
        output = self.dropout(attention_out)
        return output

class HAELT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super(HAELT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.noise_reduction = ResNetNoiseReduction(input_dim, hidden_dim // 2)
        
        self.hybrid_layers = nn.ModuleList([
            HybridLSTMTransformer(
                input_dim=input_dim if i == 0 else hidden_dim * 2,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        self.global_attention = TemporalSelfAttention(
            input_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ensemble_head = nn.ModuleList([
            nn.Linear(hidden_dim // 2, output_dim) for _ in range(3)
        ])
        
        self.final_fusion = nn.Linear(3, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_denoised = self.noise_reduction(x)
        
        hidden = x_denoised
        for hybrid_layer in self.hybrid_layers:
            hidden = hybrid_layer(hidden)
        
        attended = self.global_attention(hidden)
        
        pooled = torch.mean(attended, dim=1)
        
        features = self.feature_fusion(pooled)
        
        ensemble_outputs = []
        for head in self.ensemble_head:
            ensemble_outputs.append(head(features))
        
        ensemble_tensor = torch.cat(ensemble_outputs, dim=1)
        
        final_output = self.final_fusion(ensemble_tensor)
        
        return final_output

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

class HAELTTrainer:
    def __init__(self, model: HAELT, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.criterion = DirectionalLoss()
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
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

def create_haelt_model(input_dim: int, config: dict) -> HAELT:
    model = HAELT(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    logger.info(f"Created HAELT model with {sum(p.numel() for p in model.parameters())} parameters")
    return model
