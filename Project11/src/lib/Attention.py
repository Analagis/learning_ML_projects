# attention_machine_translation.py
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
import time

from utils_models import check_translation, timer

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
    def __init__(self, rus_vocab_size, hidden_size=64, eng_max_len=13, pos_encoding=None, max_len=21, multi_head = False, n_heads = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.eng_max_len = eng_max_len
        self.pos_encoding = pos_encoding
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding для русских букв
        self.embedding = nn.Embedding(rus_vocab_size, hidden_size)

        if multi_head:
            self.mha = MultiHeadAttention(hidden_size, hidden_size, n_heads=n_heads, device=self.device)
        
        # GRU Decoder
        self.gru = nn.GRU(2*hidden_size, hidden_size, batch_first=True)
        
        # Attention механизм
        self.attention = BahdanauAttention(hidden_size)
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, rus_vocab_size)

        self.pe = None
        if pos_encoding == 'sine':
            self.pe = self._create_sine_pe(hidden_size, max_len)
        elif pos_encoding == 'weights':
            self.pe = nn.Parameter(torch.zeros(max_len, hidden_size, device=self.device))
    
    def forward(self, decoder_input, encoder_outputs, encoder_hidden):
        """
        decoder_input: [B, rus_seq_len]
        encoder_outputs: [B, eng_seq_len, hidden_size] - ВСЕ скрытые состояния encoder'а!
        encoder_hidden: [1, B, hidden_size] - начальное состояние
        """
        # Embed decoder input
        embedded = self.embedding(decoder_input)  # [B, rus_seq_len, embed_size]

        if self.pos_encoding != 'none' and self.pe is not None:
            pe = self.pe[:decoder_input.size(1)].unsqueeze(0).to(embedded.device)
            embedded = embedded + pe
        
        # Подготавливаем для attention: decoder_hidden повторяем для всех временных шагов
        decoder_hidden_expanded = encoder_hidden[-1]  # [B, hidden_size]
        
        # Attention для КАЖДОГО временного шага decoder'а
        attention_weights, context = self.compute_attention(
            decoder_hidden_expanded, encoder_outputs
        )
        
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
        # BahdanauAttention ожидает query [B, 1, H] и keys [B, S, H]
        query = decoder_hidden.unsqueeze(1)          # [B, 1, H]

        context, attn_weights = self.attention(
            query=query,
            keys=encoder_outputs
        )
        
        attention_weights = attn_weights.squeeze(1)  # [B, eng_seq_len]
        context = context.squeeze(1)                 # [B, hidden_size]

        return attention_weights, context
    
    def _create_sine_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def translate(self, encoder, eng_name_indices, rus_char2idx, rus_idx2char, encoder_outputs=None, max_len=30):
        """Детерминированный перевод с attention"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Linear projections
        Q = self.q_linear(query)  # [B, seq_len, embed_size]
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape для multi-head: [B, n_heads, seq_len, head_dim]
        Q = Q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
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

def compute_loss(decoder, encoder, batch_X, batch_y, criterion, encoder_method='get_encoder_outputs'):
    """
    Вычисляет loss для одного батча (train/val).
    """
    encoder_outputs = encoder.get_encoder_outputs(batch_X)
    encoder_hidden = encoder.get_encoder_state(batch_X)
    
    decoder_input = batch_y[:, :-1]   # [B, rus_len-1]
    decoder_target = batch_y[:, 1:]   # [B, rus_len-1]
    
    logits, _, _ = decoder(decoder_input, encoder_outputs, encoder_hidden)
    logits = logits.reshape(-1, logits.size(-1))
    targets = decoder_target.reshape(-1)
    
    loss = criterion(logits, targets)
        
    return loss

@timer
def train_attention_decoder(encoder, train_loader, valid_loader, config, X_train_t, X_valid_t, epochs=100, lr=0.0003, patience=15, suffix="", **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = False
    
    decoder = AttentionDecoder(config['rus_vocab_size'], hidden_size=encoder.hidden_size, **kwargs).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')

    patience_counter = 0
    
    print("Training Attention Decoder...")
    
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            loss = compute_loss(decoder, encoder, batch_X, batch_y, criterion)

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
                
                loss = compute_loss(decoder, encoder, batch_X, batch_y, criterion)

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

def visualize_attention(encoder, decoder, eng_name_indices, eng_idx2char, rus_char2idx, rus_idx2char, max_len=21):
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
    input_token = torch.tensor([[sos_idx]], device=device)
    generated_tokens = [sos_idx]
    attention_weights_list = []
    
    for _ in range(max_len):
        logits, hidden, attn_weights = decoder(input_token, encoder_outputs, encoder_hidden)
        next_token = torch.argmax(logits[0, 0, :], dim=-1).item()
        
        current_weights = torch.zeros(max_len, device=device)
        actual_eng_len = len(eng_name_indices)
        if attn_weights.size(1) >= actual_eng_len:
            current_weights[:actual_eng_len] = attn_weights[0, :actual_eng_len]
        else:
            current_weights[:attn_weights.size(1)] = attn_weights[0, :]

        attention_weights_list.append(attn_weights[0].cpu().detach().numpy())  # [eng_len]
        generated_tokens.append(next_token)
        
        if next_token == rus_char2idx['>']:
            break
            
        input_token = torch.tensor([[next_token]], device=device)
        encoder_hidden = hidden
    
    # Русское имя (без SOS)
    rus_name = ''.join([rus_idx2char[idx] for idx in generated_tokens])
    eng_name = ''.join([eng_idx2char[idx] for idx in eng_name_indices])
    
    return eng_name, rus_name, attention_weights_list

