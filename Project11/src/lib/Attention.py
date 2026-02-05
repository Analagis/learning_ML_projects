# attention_machine_translation.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class AttentionDecoder(nn.Module):
    def __init__(self, rus_vocab_size, embed_size=64, hidden_size=128, eng_max_len=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.eng_max_len = eng_max_len
        
        # Embedding для русских букв
        self.embedding = nn.Embedding(rus_vocab_size, embed_size)
        
        # GRU Decoder
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        
        # Attention механизм (ручной!)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_combine = nn.Linear(hidden_size + hidden_size, hidden_size)
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, rus_vocab_size)
    
    def forward(self, decoder_input, encoder_outputs, encoder_hidden):
        """
        decoder_input: [B, rus_seq_len]
        encoder_outputs: [B, eng_seq_len, hidden_size] - ВСЕ скрытые состояния encoder'а!
        encoder_hidden: [1, B, hidden_size] - начальное состояние
        """
        B = decoder_input.size(0)
        
        # Embed decoder input
        embedded = self.embedding(decoder_input)  # [B, rus_seq_len, embed_size]
        
        # Подготавливаем для attention: decoder_hidden повторяем для всех временных шагов
        decoder_hidden_expanded = encoder_hidden[-1]  # [B, hidden_size]
        
        # Attention для КАЖДОГО временного шага decoder'а
        attention_weights = self.compute_attention(
            decoder_hidden_expanded, encoder_outputs
        )  # [B, eng_seq_len]
        
        # Context vector [B, hidden_size]
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, eng_seq_len]
            encoder_outputs  # [B, eng_seq_len, hidden_size]
        ).squeeze(1)  # [B, hidden_size]
        
        # Конкатенируем embedded + context
        gru_input = torch.cat([embedded, context.unsqueeze(1).expand(-1, embedded.size(1), -1)], dim=2)
        
        # GRU forward
        gru_output, hidden = self.gru(gru_input, encoder_hidden)
        
        # Выход
        logits = self.fc_out(gru_output)  # [B, rus_seq_len, rus_vocab]
        return logits, hidden, attention_weights
    
    def compute_attention(self, decoder_hidden, encoder_outputs):
        """
        Вычисляет attention weights между decoder_hidden и encoder_outputs
        decoder_hidden: [B, hidden_size]
        encoder_outputs: [B, eng_seq_len, hidden_size]
        """
        # Attention energy: decoder_hidden * encoder_outputs
        energy = torch.tanh(self.attention(
            torch.cat([decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1), 
                      encoder_outputs], dim=2)
        ))  # [B, eng_seq_len, hidden_size]
        
        attention_scores = torch.sum(energy, dim=2)  # [B, eng_seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, eng_seq_len]
        return attention_weights
    
    def translate(self, encoder, eng_name_indices, rus_char2idx, rus_idx2char, encoder_outputs=None, max_len=30):
        """Детерминированный перевод с attention"""
        device = next(self.parameters()).device
        
        # Encode
        eng_tensor = torch.tensor([eng_name_indices], dtype=torch.long, device=device)
        encoder_hidden = encoder.get_encoder_state(eng_tensor)  # [1, 1, hidden]
        
        # Получаем encoder_outputs (ВСЕ состояния!)
        with torch.no_grad():
            embedded = encoder.embedding(eng_tensor)
            encoder_outputs_full, _ = encoder.gru(embedded)
        
        # Decoder
        sos_idx = rus_char2idx['<']
        input_token = torch.tensor([[sos_idx]], device=device)
        generated = [sos_idx]
        
        for _ in range(max_len):
            logits, hidden, attention_weights = self(
                input_token, encoder_outputs_full, encoder_hidden
            )
            next_token = torch.argmax(logits[0, 0, :], dim=-1).item()
            
            if next_token == rus_char2idx['>']:
                break
                
            generated.append(next_token)
            input_token = torch.tensor([[next_token]], device=device)
            encoder_hidden = hidden
        
        translation = ''.join([rus_idx2char[idx] for idx in generated[1:] 
                             if idx != rus_char2idx['.']])
        return translation

def extract_encoder_outputs(encoder, batch_X):
    """Извлекает ВСЕ скрытые состояния encoder'а"""
    embedded = encoder.embedding(batch_X)
    encoder_outputs, _ = encoder.gru(embedded)  # [B, eng_seq_len, hidden]
    return encoder_outputs

def train_attention_decoder(encoder, train_loader, valid_loader, rus_vocab_size, eng_idx2char, rus_idx2char, 
                          eng_char2idx, rus_char2idx, X_valid_t, y_valid_t, epochs=100, lr=0.0003, patience=15):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = False
    
    decoder = AttentionDecoder(rus_vocab_size, hidden_size=encoder.hidden_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=rus_char2idx['.'])
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    
    print("Training Attention Decoder...")
    
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Encoder outputs + initial hidden
            encoder_outputs = extract_encoder_outputs(encoder, batch_X)  # [B, eng_len, H]
            encoder_hidden = encoder.get_encoder_state(batch_X)          # [1, B, H]
            
            # Decoder forward
            decoder_input = batch_y[:, :-1]   # [B, rus_len-1]
            decoder_target = batch_y[:, 1:]   # [B, rus_len-1]
            
            logits, _, _ = decoder(decoder_input, encoder_outputs, encoder_hidden)
            logits = logits.reshape(-1, logits.size(-1))
            targets = decoder_target.reshape(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        # Validation
        decoder.eval()
        encoder.eval()
        valid_loss = 0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                encoder_outputs = extract_encoder_outputs(encoder, batch_X)
                encoder_hidden = encoder.get_encoder_state(batch_X)
                
                decoder_input = batch_y[:, :-1]
                decoder_target = batch_y[:, 1:]
                
                logits, _, _ = decoder(decoder_input, encoder_outputs, encoder_hidden)
                logits = logits.reshape(-1, logits.size(-1))
                targets = decoder_target.reshape(-1)
                
                loss = criterion(logits, targets)
                valid_loss += loss.item()
                num_valid_batches += 1
        
        avg_train_loss = train_loss / num_batches
        avg_valid_loss = valid_loss / num_valid_batches
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        print(f'Epoch {epoch+1}: Train={avg_train_loss:.4f}, Valid={avg_valid_loss:.4f}')
        
        # Early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(decoder.state_dict(), 'best_attention_decoder.pth')
    
    decoder.load_state_dict(torch.load('best_attention_decoder.pth'))
    return decoder, train_losses, valid_losses
