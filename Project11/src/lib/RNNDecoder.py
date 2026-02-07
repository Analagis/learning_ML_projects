import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from RNNEncoder import RNNEncoder



class RNNDecoder(nn.Module):
    def __init__(self, rus_vocab_size, embed_size=64, hidden_size=64, pos_encoding=None, max_len=21):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_encoding = pos_encoding
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding для русских букв
        self.embedding = nn.Embedding(rus_vocab_size, embed_size)
        
        # GRU Decoder
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, rus_vocab_size)

        self.pe = None
        if pos_encoding == 'sine':
            self.register_buffer('pe', self._create_sine_pe(embed_size, max_len))
        elif pos_encoding == 'trainable':
            self.pe = nn.Parameter(torch.zeros(max_len, embed_size, device=self.device))
    
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

def compute_perplexity(model, encoder, test_loader, rus_char2idx):
    """Perplexity = exp(average NLL loss)"""
    device = next(model.parameters()).device
    model.eval()
    encoder.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=rus_char2idx['.'], reduction='sum')
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Encoder hidden
            encoder_hidden = encoder.get_encoder_state(batch_X)
            
            # Decoder шаг за шагом
            input_token = torch.full((batch_y.size(0), 1), rus_char2idx['<'], device=device)
            decoder_hidden = encoder_hidden
            
            for t in range(1, batch_y.size(1)):
                logits, decoder_hidden = model(input_token, decoder_hidden)
                target = batch_y[:, t:t+1].reshape(-1)
                input_token = batch_y[:, t-1:t].reshape(-1, 1)  
                
                loss = criterion(logits.reshape(-1, logits.size(-1)), target)
                total_loss += loss.item()
                total_tokens += (target != rus_char2idx['.']).sum().item()
    
    avg_nll = total_loss / total_tokens
    perplexity = np.exp(avg_nll)
    return perplexity

def train_decoder(encoder, train_loader, valid_loader, rus_vocab_size, eng_idx2char, rus_idx2char, eng_char2idx, rus_char2idx, 
                  X_train_t, X_valid_t, max_len, epochs=100, lr=0.001, patience=15, embed_size=128, hidden_size=256, suffix=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to()
    encoder.eval()  # Важно: eval режим для энкодера!
    for param in encoder.parameters(): # Замораживаем веса
        param.requires_grad = False
    
    decoder = RNNDecoder(rus_vocab_size, embed_size, hidden_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=rus_char2idx['.'])
    optimizer = optim.Adam(decoder.parameters(), lr=lr)  # Только decoder
    
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience_counter = 0
    
    print("Training Decoder...")
    
    # Получаем индексы SOS и EOS для teacher forcing
    sos_idx = rus_char2idx['<']
    
    for epoch in range(epochs):
        # === TRAIN ===
        decoder.train()
        train_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            batch_X = batch_X.to(device)  # English [batch, seq_len]
            batch_y = batch_y.to(device)  # Russian [batch, seq_len]
            
            optimizer.zero_grad()
            
            batch_size = batch_X.size(0)
            max_target_len = batch_y.size(1) - 1  # -1 потому что начинаем с SOS
            
            # 1. Получаем скрытое состояние от энкодера
            encoder_hidden = encoder.get_encoder_state(batch_X)
            
            # 2. Начинаем с SOS токена для всех в батче
            decoder_input = torch.full((batch_size, 1), sos_idx, device=device)
            decoder_hidden = encoder_hidden
            
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            step_count = 0
            
            # 3. Проходим по каждому шагу последовательности
            for t in range(max_target_len):
                # Прямой проход через декодер (один шаг)
                logits, decoder_hidden = decoder(decoder_input, decoder_hidden)
                # logits: [batch_size, 1, vocab_size]
                
                # Целевой токен для этого шага (без SOS)
                target_token = batch_y[:, t + 1]  # Сдвиг на 1, так как первый токен - SOS
                # target_token: [batch_size]

                pad_mask = (target_token == rus_char2idx['.'])
                if pad_mask.all():
                    continue
                
                # Вычисляем потерю для этого шага
                step_loss = criterion(logits.squeeze(1), target_token)
                
                total_loss = total_loss + step_loss
                step_count += 1
                
                # Teacher forcing: берем правильный следующий токен как вход для следующего шага
                # (но не на последнем шаге)
                decoder_input = batch_y[:, t + 1].unsqueeze(1)
                    
            if step_count:
                # 4. Вычисляем среднюю потерю
                avg_loss = total_loss / step_count
                avg_loss.backward()
                
                # Обрезаем градиенты
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
                optimizer.step()
                
                train_loss += avg_loss.item()
            num_batches += 1
        
        # === VALID ===
        decoder.eval()
        valid_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Аналогичный пошаговый процесс для валидации
                batch_size = batch_X.size(0)
                max_target_len = batch_y.size(1) - 1
                
                encoder_hidden = encoder.get_encoder_state(batch_X)
                decoder_input = torch.full((batch_size, 1), sos_idx, device=device)
                decoder_hidden = encoder_hidden
                
                total_val_loss = 0
                step_count = 0
                
                for t in range(max_target_len):
                    logits, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    target_token = batch_y[:, t + 1]
                    pad_mask = (target_token == rus_char2idx['.'])
                    if pad_mask.all():
                        continue
                    
                    step_loss = criterion(logits.squeeze(1), target_token)
                    total_val_loss += step_loss
                    step_count += 1
                    
                    if t < max_target_len - 1:
                        decoder_input = batch_y[:, t + 1].unsqueeze(1)
                    else:
                        decoder_input = torch.argmax(logits, dim=-1)
                
                valid_loss += (total_val_loss / step_count).item()
                num_valid_batches += 1
        
        avg_train_loss = train_loss / num_batches
        avg_valid_loss = valid_loss / num_valid_batches
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        print(f'Epoch {epoch+1}: Train={avg_train_loss:.4f}, Valid={avg_valid_loss:.4f}')

         # Печать перевода
        if (epoch + 1) % (epochs // 4) == 0 or epoch == epochs - 1:
            print("=== TRAIN SET ===")
            encoder.eval()
            decoder.eval()
            
            # Берем первые 5 примеров из train
            check_translation(X_train_t[:5], eng_char2idx, eng_idx2char, decoder, encoder, rus_char2idx, rus_idx2char, max_len, n=5)
            
            print("\n=== VALID SET ===")
            check_translation(X_valid_t[:5], eng_char2idx, eng_idx2char, decoder, encoder, rus_char2idx, rus_idx2char, max_len, n=5)
            print("-" * 50)
            
            # Возвращаем в train режим
            decoder.train()
            encoder.eval()
       
        # Early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), f'best_decoder_{suffix}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Загружаем лучшую модель
    decoder.load_state_dict(torch.load(f'best_decoder_{suffix}.pth'))
    return decoder, train_losses, valid_losses


def check_translation(X_test_t, eng_char2idx, eng_idx2char, decoder, encoder, rus_char2idx, rus_idx2char, max_len, n):
    translations = []
    for i in range(n):
        eng_indices = [idx.item() for idx in X_test_t[i] if idx.item() not in [eng_char2idx['.'], eng_char2idx['<'], eng_char2idx['>']]]
        eng_name = ''.join([eng_idx2char[idx] for idx in eng_indices])
        rus_pred = decoder.translate(encoder, eng_indices, rus_char2idx, rus_idx2char, max_len)
        translations.append(f"{eng_name:10s}→{rus_pred}")
    print(" | ".join(translations))
        