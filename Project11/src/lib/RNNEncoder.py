import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time

from utils_models import timer

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=64, num_layers=1, pos_encoding=None, max_len=13):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.pos_encoding = pos_encoding  # 'none', 'sine', 'trainable'
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding: символ -> вектор
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # GRU: последовательность символов -> скрытое состояние
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear: скрытое состояние -> вероятности букв (для языковой модели)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.pe = None
        if pos_encoding == 'sine':
            self.pe = self._create_sine_pe()
        elif pos_encoding == 'trainable':
            self.pe = nn.Parameter(torch.zeros(max_len, embed_size, device=self.device))
        
    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_size]
        if self.pe is not None:
            embedded = embedded + self.pe[:x.size(1)].unsqueeze(0).to(x.device)

        output, hidden = self.gru(embedded, hidden)  # [batch, seq_len, hidden], hidden: [1, batch, hidden]
        logits = self.fc_out(output)  # [batch, seq_len, vocab_size]
        return logits, hidden
    
    def get_encoder_state(self, x):
        embedded = self.embedding(x)

        if self.pe is not None:
            embedded = embedded + self.pe[:x.size(1)].unsqueeze(0).to(x.device)

        _, hidden = self.gru(embedded)
        return hidden  # Финальное скрытое состояние [1, batch, hidden_size]
    
    def get_encoder_outputs(self, x):
        """Извлекает ВСЕ скрытые состояния encoder'а"""
        embedded = self.embedding(x)

        if self.pe is not None:
            embedded = embedded + self.pe[:x.size(1)].unsqueeze(0).to(x.device)

        encoder_outputs, _ = self.gru(embedded)  # [B, eng_seq_len, hidden]
        return encoder_outputs
    
    def _create_sine_pe(self):
        """Sine/Cosine positional encoding"""
        pe = torch.zeros(self.max_len, self.embed_size)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * 
                            -(torch.log(torch.tensor(10000.0)) / self.embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
def compute_loss_batch(model, batch_X, criterion):
    """Вычисляет loss для одного батча language modeling"""
    logits, _ = model(batch_X)
    target_tokens = batch_X[:, 1:]
    
    logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    targets = target_tokens.reshape(-1)
    
    return criterion(logits, targets)    

# --- Training Loop с Early Stopping ---
@timer
def train_rnn_encoder(model, train_loader, valid_loader, pad_idx, epochs=100, lr=0.001, patience=10, suffix=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Игнорируем PAD
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            
            optimizer.zero_grad()

            loss = compute_loss_batch(model, batch_X, criterion)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_X, _ in valid_loader:
                batch_X = batch_X.to(device)

                valid_loss += compute_loss_batch(model, batch_X, criterion)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        if (epoch + 1) % (epochs // 4) == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}: Train={avg_train_loss:.4f}, Valid={avg_valid_loss:.4f}')
        
        # Early Stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_models/best_encoder{suffix}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load(f'best_models/best_encoder{suffix}.pth'))
    
    return train_losses, valid_losses

# --- Генерация имен ---
@torch.no_grad()
def generate_names(model, sos_idx, eos_idx, pad_idx, eng_idx2char, num_names=10, max_len=20, temperature=0.8):
    device = next(model.parameters()).device
    model.eval()
    
    names = []
    hidden_size = model.hidden_size
    
    for _ in range(num_names):
        # ✅ ПРАВИЛЬНО: создаем нулевое hidden state
        batch_size = 1
        num_layers = 1  # у нас 1-layer GRU
        hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        
        # Старт с SOS
        input_seq = torch.tensor([[sos_idx]], device=device)  # [1, 1]
        generated = []
        
        for _ in range(max_len):
            logits, hidden = model(input_seq, hidden)  # hidden обновляется!
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == pad_idx or next_token == sos_idx:
                next_token = eos_idx

            if next_token == eos_idx:
                break
                
            generated.append(next_token)
            input_seq = torch.tensor([[next_token]], device=device)
        
        name = ''.join([eng_idx2char[idx] for idx in generated])
        names.append(name)
    
    return names

# Загружаем обученный Encoder
def load_encoder(vocab_size, checkpoint_path='best_models/best_encoder.pth', embed_size=128, hidden_size=256):
    """Загружает обученный Encoder"""
    encoder = RNNEncoder(vocab_size, embed_size, hidden_size)
    
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint)
    encoder.eval()  # Freeze!
    
    print(f"✅ Encoder загружен")
    return encoder