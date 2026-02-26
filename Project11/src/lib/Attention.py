import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import math
import random
import time

from utils_models import timer

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class AttentionDecoder(nn.Module):
    def __init__(self, rus_vocab_size, hidden_size=64, eng_max_len=13, pos_encoding=None, max_len=15, multi_head = False, n_heads = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.eng_max_len = eng_max_len
        self.pos_encoding = pos_encoding
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_head = multi_head
        
        # Embedding для русских букв
        self.embedding = nn.Embedding(rus_vocab_size, hidden_size)

        if self.multi_head:
            self.mha = MultiHeadAttention(hidden_size, hidden_size, n_heads=n_heads, device=self.device)
        
        # GRU Decoder
        self.gru = nn.GRU(2*hidden_size, hidden_size, batch_first=True)
        
        # Attention механизм
        self.attention = BahdanauAttention(hidden_size)
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, rus_vocab_size)

        self.pe = None
        if pos_encoding is not None:
            # Sinusoidal PE как в Transformer
            pe = torch.zeros(hidden_size, hidden_size)
            position = torch.arange(0, hidden_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            if pos_encoding == 'sine':
                self.pe = pe.unsqueeze(0)
            else:
                self.pe = nn.Embedding(hidden_size, hidden_size)
                with torch.no_grad():
                    self.pe.weight.copy_(pe)
    
    def forward(self, decoder_input, encoder_outputs, encoder_hidden):
        """
        decoder_input: [B]
        encoder_outputs: [B, eng_seq_len, hidden_size] - ВСЕ скрытые состояния encoder'а!
        encoder_hidden: [1, B, hidden_size] - начальное состояние
        """
        # Embed decoder input
        embedded = self.embedding(decoder_input).unsqueeze(1)  # [B, rus_seq_len, embed_size]

        if self.pos_encoding == 'sine' and self.pe is not None:
            pe = self.pe[:, :1].to(embedded.device)
            embedded = embedded + pe
            
        if self.pos_encoding == 'weights' and self.pe is not None:
            # Создаём позиции [0,1,2,...] для текущей длины
            positions = torch.zeros(1, dtype=torch.long,
                                device=decoder_input.device)  # [1]
            pe = self.pe(positions).unsqueeze(1)  # [1, seq_len, hidden_size]
            pe = F.dropout(pe, p=0.1, training=self.training)
            embedded = embedded + pe
        
        # Подготавливаем для attention: decoder_hidden повторяем для всех временных шагов
        decoder_hidden_expanded = encoder_hidden[-1].unsqueeze(1)  # [B, 1, H]
        
        if self.multi_head:
            context, attention_weights = self.mha(
                decoder_hidden_expanded,          # [B,1,H]
                encoder_outputs,  # [B,S,H]
                encoder_outputs   # [B,S,H]
            )
        else:
            context, attention_weights = self.attention(
                decoder_hidden_expanded, encoder_outputs
            )
        
        # Конкатенируем embedded + context
        gru_input = torch.cat([embedded, context], dim=2)
        
        # GRU forward
        gru_output, hidden = self.gru(gru_input, encoder_hidden)
        
        # Выход
        logits = self.fc_out(gru_output.squeeze(1))  # [B, rus_seq_len, rus_vocab]

        if self.multi_head:
            attention_weights = attention_weights.squeeze(2)
        else:
            attention_weights = attention_weights.squeeze(1)
        return logits, hidden, attention_weights
    
    def _create_sine_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, n_heads=3, dropout=0.1, device='cpu'):
        super().__init__()
        assert embed_size % n_heads == 0
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        self.device = device
        
        # Linear projections для Q, K, V
        self.q_linear = nn.Linear(hidden_size, embed_size)
        self.k_linear = nn.Linear(hidden_size, embed_size)
        self.v_linear = nn.Linear(hidden_size, embed_size)
        
        # Output projection
        self.out_linear = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        """
        query/key/value: [B, seq_len, hidden_size]
        """
        B, seq_len, _ = query.shape
        _, k_len, _ = key.shape
        
        # Linear projections
        Q = self.q_linear(query)  # [B, seq_len, embed_size]
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape для multi-head: [B, n_heads, seq_len, head_dim]
        Q = Q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores [B, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax по последнему измерению
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        context = torch.matmul(attn_weights, V)  # [B, n_heads, seq_len, head_dim]
        
        # Concat heads и final projection
        context = context.transpose(1, 2).contiguous().view(B, seq_len, self.embed_size)
        output = self.out_linear(context)
        
        return output, attn_weights

def compute_loss(decoder, encoder, batch_X, batch_y, criterion, config, tf_ratio = 1):
    """Teacher forcing ПОШАГОВО для attention decoder"""
    device = batch_X.device
    batch_size = batch_X.size(0)
    
    # Encode
    encoder_outputs = encoder.get_encoder_outputs(batch_X)  # [B, eng_len, H]
    encoder_hidden = encoder.get_encoder_state(batch_X)     # [1, B, H]
    
    decoder_input = torch.full((batch_size,), config['sos_idx'], device=device, dtype=torch.long)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    step_count = 0
    
    # ПОШАГОВО через target sequence
    for t in range(1, batch_y.size(1)):
        target_token = batch_y[:, t]  # [B]

        if (target_token == config['pad_idx']).all():  # токены в батче PAD
            break

        # Forward ОДНОГО шага
        logits, decoder_hidden, _ = decoder(
            decoder_input, encoder_outputs, encoder_hidden
        )  # logits: [B, vocab]
        
        # Loss для ЭТОГО шага
        step_loss = criterion(logits, target_token)
        
        if torch.isnan(step_loss) or torch.isinf(step_loss):
            continue

        total_loss = total_loss + step_loss
        step_count += 1
        
        use_teacher_forcing = random.random() < tf_ratio

        if use_teacher_forcing:
            decoder_input = target_token              # ground truth
        else:
            decoder_input = torch.argmax(logits, dim=-1)  # модельное предсказание

        encoder_hidden = decoder_hidden  # обновляем hidden!
       
    return total_loss / step_count if step_count > 0 else torch.tensor(0.0, device=device)

@timer
def train_attention_decoder(encoder, train_loader, valid_loader, config, X_train_t, X_valid_t, epochs=100, lr=0.001, patience=15, suffix="", teacher_forcing_ratio=1.0, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = False
    
    decoder = AttentionDecoder(config['rus_vocab_size'], hidden_size=encoder.hidden_size, **kwargs).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-4)
    
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')

    patience_counter = 0
    
    print("Training Attention Decoder...")
    
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0
        num_batches = 0
        tf_ratio = max(0.0, teacher_forcing_ratio * (1 - epoch / epochs))
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            loss = compute_loss(decoder, encoder, batch_X, batch_y, criterion, config, tf_ratio)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
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
                
                loss = compute_loss(decoder, encoder, batch_X, batch_y, criterion, config)

                valid_loss += loss.item()
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
            check_translation(X_train_t[:5], config['eng_idx2char'], decoder, encoder, config['rus_char2idx'], config['rus_idx2char'], config['y_max_len'], n=5)
            
            print("=== VALID SET ===")
            check_translation(X_valid_t[:5], config['eng_idx2char'], decoder, encoder, config['rus_char2idx'], config['rus_idx2char'], config['y_max_len'], n=5)
            print("-" * 50)
            
            # Возвращаем в train режим
            decoder.train()
            encoder.eval()
        
        # Early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), f'best_models/best_attention_decoder{suffix}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    decoder.load_state_dict(torch.load(f'best_models/best_attention_decoder{suffix}.pth'))
    return decoder, train_losses, valid_losses

def plot_attention_heatmap(eng_name, rus_name, attention_weights_list, max_length=23):
    """
    eng_name: исходное имя с < и >
    rus_name: переведенное имя с < и >
    attention_weights_list: список весов внимания для каждого шага декодера
    """
    
    eng_letters = list(eng_name)
    rus_letters = list(rus_name)
    
    # Определяем максимальную длину для паддинга
    max_eng_len = max_length
    max_rus_len = max_length
    
    # Создаем матрицу внимания с паддингом
    attention_matrix = np.zeros((max_rus_len, max_eng_len))
    
    # Заполняем актуальными значениями
    for i in range(len(rus_letters)):
        if i < len(attention_weights_list):
            # Берем веса для текущего шага декодера
            step_weights = attention_weights_list[i]
            
            # Обрезаем или дополняем веса до max_eng_len
            if len(step_weights) > max_eng_len:
                attention_matrix[i, :] = step_weights[:max_eng_len]
            else:
                attention_matrix[i, :len(step_weights)] = step_weights
                # Остальное остается 0 (паддинг)
    
    # Создаем метки с паддингом
    eng_labels = eng_letters + [''] * (max_rus_len - len(eng_letters))
    rus_labels = rus_letters + [''] * (max_rus_len - len(rus_letters))
    
    # Создаем фигуру с верхними метками для английских букв
    fig, ax = plt.subplots(figsize=(max_rus_len*0.2, max_rus_len*0.2))
    
    # Отображаем heatmap
    im = ax.imshow(attention_matrix.T, 
                   cmap='viridis', aspect='auto')
    
    # Устанавливаем метки
    ax.set_yticks(range(max_rus_len))
    ax.set_yticklabels(eng_labels)
    ax.set_ylabel('English letters (Source)')
    
    ax.set_xticks(range(max_rus_len))
    ax.set_xticklabels(rus_labels)
    ax.set_xlabel('Russian letters (Target)')
    
    # Добавляем сверху название перевода
    plt.title(f'Attention Map: "{eng_name}" → "{rus_name}"', pad=20)
    
    # Добавляем цветовую шкалу
    plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.show()

def translate(encoder, decoder, eng_name_indices, eng_idx2char, rus_char2idx, rus_idx2char, max_len=15, head = None):
    """Визуализация attention weights для одного имени"""
    device = next(decoder.parameters()).device
    encoder.eval()
    decoder.eval()
    
    # Encode
    eng_tensor = torch.tensor([eng_name_indices], dtype=torch.long, device=device)
    embedded = encoder.embedding(eng_tensor)
    encoder_outputs, encoder_hidden = encoder.gru(embedded)
    
    # Генерируем перевод
    sos_idx = rus_char2idx['<']
    input_token = torch.tensor([sos_idx], device=device)
    generated_tokens = [sos_idx]
    attention_weights_list = []
    
    for _ in range(max_len):
        logits, hidden, attn_weights = decoder(input_token, encoder_outputs, encoder_hidden)
        next_token = torch.argmax(logits[0], dim=-1).item()
        
        if head is None:
            attn_weights = attn_weights[0]
        else:
            attn_weights = attn_weights[0,head]
        attention_weights_list.append(attn_weights.cpu().detach().numpy())  # [eng_len]
        generated_tokens.append(next_token)
        
        if next_token == rus_char2idx['>']:
            break
            
        input_token = torch.tensor([next_token], device=device)    # [1]
        encoder_hidden = hidden
    
    # Русское имя (без SOS)
    rus_name = ''.join([rus_idx2char[idx] for idx in generated_tokens])
    eng_name = ''.join([eng_idx2char[idx] for idx in eng_name_indices])
    
    return eng_name, rus_name, attention_weights_list

def check_translation(X_test_t, eng_idx2char, decoder, encoder, 
                     rus_char2idx, rus_idx2char, y_max_len, n):
    translations = []
    for i in range(n):
        eng_indices = [idx.item() for idx in X_test_t[i]]
        eng_name, rus_name, _ = translate(encoder, decoder, eng_indices, eng_idx2char, rus_char2idx, rus_idx2char, max_len=y_max_len)
        translations.append(f"{eng_name.replace('.', ''):10s}→{rus_name}")
    print(" | ".join(translations))
    