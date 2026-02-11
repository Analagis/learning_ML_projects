import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import time

from utils_models import check_translation, timer



class RNNDecoder(nn.Module):
    def __init__(self, rus_vocab_size, hidden_size=64, pos_encoding=None, max_len=21):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_encoding = pos_encoding
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding для русских букв
        self.embedding = nn.Embedding(rus_vocab_size, hidden_size)
        
        # GRU Decoder
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, rus_vocab_size)

        self.pe = None
        if pos_encoding == 'sine':
            self.pe = self._create_sine_pe(hidden_size, max_len)
        elif pos_encoding == 'weights':
            self.pe = nn.Parameter(torch.zeros(max_len, hidden_size, device=self.device))
    
    def forward(self, x, hidden):

        embedded = self.embedding(x)  

        if self.pe is not None:
            embedded = embedded + self.pe[:x.size(1)].unsqueeze(0).to(x.device)

        output, hidden = self.gru(embedded, hidden)  
        logits = self.fc_out(output) 
        return logits, hidden
    
    def _create_sine_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def translate(self, encoder, eng_name_indices, rus_char2idx, rus_idx2char, max_len):
        """Детерминированный перевод (argmax, не стохастический)"""
        
        # Encode английское имя
        eng_tensor = torch.tensor([eng_name_indices], dtype=torch.long, device=self.device)
        encoder_hidden = encoder.get_encoder_state(eng_tensor)  # [1, 1, hidden_size]

        if len(encoder_hidden.shape) == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        
        # Decoder начинает с <SOS>
        sos_idx = rus_char2idx['<']
        input_token = torch.tensor([[sos_idx]], device=self.device)
        
        # Инициализируем скрытое состояние для декодера
        decoder_hidden = encoder_hidden  # Начальное состояние = выход энкодера
        
        translated_indices = [sos_idx]
        for _ in range(max_len):
            logits, decoder_hidden = self(input_token, decoder_hidden)  # Используем decoder_hidden!
            next_token = torch.argmax(logits[0, 0, :], dim=-1).item()
            
            if next_token == rus_char2idx['>']:  # EOS
                break
                
            translated_indices.append(next_token)
            input_token = torch.tensor([[next_token]], device=self.device)
        
        # Убираем SOS
        translation = ''.join([rus_idx2char[idx] for idx in translated_indices[1:]])
        return translation

def train_decoder_step(decoder, encoder, batch_X, batch_y, pad_idx, criterion, sos_idx, device):
    """Один шаг teacher forcing для train/val"""
    batch_size = batch_X.size(0)
    max_target_len = batch_y.size(1) - 1
    
    encoder_hidden = encoder.get_encoder_state(batch_X)
    decoder_input = torch.full((batch_size, 1), sos_idx, device=device)
    decoder_hidden = encoder_hidden
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    step_count = 0
    
    for t in range(max_target_len):
        logits, decoder_hidden = decoder(decoder_input, decoder_hidden)
        target_token = batch_y[:, t + 1]
        
        pad_mask = (target_token == pad_idx)
        if pad_mask.all():
            continue
            
        step_loss = criterion(logits.squeeze(1), target_token)
        total_loss = total_loss + step_loss
        step_count += 1
        
        decoder_input = batch_y[:, t + 1].unsqueeze(1)
    
    return total_loss / step_count if step_count > 0 else torch.tensor(0.0, device=device, requires_grad=True)

@timer
def train_decoder(encoder, train_loader, valid_loader, config,
                  X_train_t, X_valid_t, epochs=100, lr=0.001, patience=15, suffix="", **kwargs):
    device = encoder.device
    encoder.to()
    encoder.eval()  # Важно: eval режим для энкодера!
    for param in encoder.parameters(): # Замораживаем веса
        param.requires_grad = False
    
    decoder = RNNDecoder(config['rus_vocab_size'], **kwargs).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    optimizer = optim.Adam(decoder.parameters(), lr=lr)  # Только decoder
    
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience_counter = 0
    
    print("Training Decoder...")
    
    for epoch in range(epochs):
        # === TRAIN ===
        decoder.train()
        train_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)  # English [batch, seq_len]
            batch_y = batch_y.to(device)  # Russian [batch, seq_len]
            
            optimizer.zero_grad()
            
            step_loss = train_decoder_step(decoder, encoder, batch_X, batch_y, config['pad_idx'], 
                                         criterion, config['sos_idx'], device)
            
            if step_loss.item() > 0:
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
                optimizer.step()
                train_loss += step_loss.item()
                num_batches += 1
        
        # === VALID ===
        decoder.eval()
        valid_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                step_loss = train_decoder_step(decoder, encoder, batch_X, batch_y, config['pad_idx'], 
                                             criterion, config['sos_idx'], device)
                valid_loss += step_loss
                num_valid_batches += 1
        
        avg_train_loss = train_loss / num_batches
        avg_valid_loss = valid_loss / num_valid_batches
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
         # Печать перевода
        if epoch%(patience//3) == 0 or epoch == epochs - 1:
            print(f'\033[92mEpoch {epoch+1}:\033[0m Train={avg_train_loss:.4f}, Valid={avg_valid_loss:.4f}')

            print("=== TRAIN SET ===")
            encoder.eval()
            decoder.eval()
            
            # Берем первые 5 примеров из train
            check_translation(X_train_t[:5], config['eng_char2idx'], config['eng_idx2char'], decoder, encoder, config['rus_char2idx'], config['rus_idx2char'], config['y_max_len'], n=5)
            
            print("=== VALID SET ===")
            check_translation(X_valid_t[:5], config['eng_char2idx'], config['eng_idx2char'], decoder, encoder, config['rus_char2idx'], config['rus_idx2char'], config['y_max_len'], n=5)
            print("-" * 50)
            
            # Возвращаем в train режим
            decoder.train()
            encoder.eval()
       
        # Early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), f'best_models/best_decoder{suffix}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Загружаем лучшую модель
    decoder.load_state_dict(torch.load(f'best_models/best_decoder{suffix}.pth'))
    return decoder, train_losses, valid_losses



