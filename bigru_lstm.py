import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Add positional information for anatomical slice ordering"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :x.size(2)]

class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence dimension"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        # Compute attention weights
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        weighted = x * attn_weights  # [batch, seq_len, hidden_dim]
        pooled = torch.sum(weighted, dim=1)  # [batch, hidden_dim]
        
        return pooled, attn_weights.squeeze(-1)  # Return weights for visualization

class AgePredictionCNN(nn.Module):
    """
    Drop-in replacement BiGRU model that maintains the same interface as the original CNN.
    Renamed to AgePredictionCNN for compatibility with existing code.
    """
    def __init__(self, input_shape, hidden_dim=256, num_layers=2, dropout=0.3):
        super(AgePredictionCNN, self).__init__()
        
        print("AgePredictionCNN Shape:", input_shape)
        
        # Input shape: (1, num_slices, embedding_dim)
        self.num_slices = input_shape[1]  # Should be 160
        self.embedding_dim = input_shape[2]  # Should be 768
        
        print(f"BiGRU Model - Slices: {self.num_slices}, Embedding dim: {self.embedding_dim}")
        
        # Input projection to match hidden dimension
        self.input_projection = nn.Linear(self.embedding_dim, hidden_dim)
        
        # Positional encoding for anatomical ordering
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=self.num_slices)
        
        # Bidirectional GRU layers
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_dim if i == 0 else hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < num_layers - 1 else 0
            ) for i in range(num_layers)
        ])
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_dim * 2)  # *2 for bidirectional
        
        # Classification head - matching original structure
        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc1_bn = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(129, 1)  # 128 + 1 for sex input
        self.silu = nn.SiLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize GRU weights properly"""
        for layer in self.gru_layers:
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, x, sex):
        # Input shape: [batch, 1, num_slices, embedding_dim]
        batch_size = x.size(0)
        
        # Remove the channel dimension and rearrange
        x = x.squeeze(1)  # [batch, num_slices, embedding_dim]
        
        # Project to hidden dimension
        x = self.input_projection(x)  # [batch, num_slices, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through BiGRU layers
        for gru_layer in self.gru_layers:
            x, _ = gru_layer(x)  # [batch, num_slices, hidden_dim * 2]
        
        # Attention pooling over sequence dimension
        x, attention_weights = self.attention_pool(x)  # [batch, hidden_dim * 2]
        
        # Follow the original CNN's classification head structure
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.silu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.silu(x)
        x = self.dropout(x)
        
        # Concatenate `Sex` input - matching original format
        x = torch.cat((x, sex.unsqueeze(1)), dim=1)
        x = self.fc3(x)
        
        return x
    
    def get_attention_weights(self, x, sex):
        """Get attention weights for visualization (additional method)"""
        with torch.no_grad():
            batch_size = x.size(0)
            x = x.squeeze(1)
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            
            for gru_layer in self.gru_layers:
                x, _ = gru_layer(x)
            
            _, attention_weights = self.attention_pool(x)
            return attention_weights  # [batch, num_slices]
