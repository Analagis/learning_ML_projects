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

class AttentionDecoder(nn.Module):
    def __init__(self, rus_vocab_size, embed_size=64, hidden_size=64, eng_max_len=13, pos_encoding=None, max_len=21, multi_head = False, n_heads = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.eng_max_len = eng_max_len
        self.pos_encoding = pos_encoding
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding для русских букв
        self.embedding = nn.Embedding(rus_vocab_size, embed_size)

        if multi_head:
            self.mha = MultiHeadAttention(embed_size, hidden_size, n_heads=n_heads, device=self.device)
        
        # GRU Decoder
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        
        # Attention механизм (ручной!)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_combine = nn.Linear(hidden_size + hidden_size, hidden_size)
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, rus_vocab_size)

        self.pe = None
        if pos_encoding == 'sine':
            self.register_buffer('pe', self._create_sine_pe(embed_size, max_len))
        elif pos_encoding == 'trainable':
            self.pe = nn.Parameter(torch.zeros(max_len, embed_size, device=self.device))
    
    def forward(self, decoder_input, encoder_outputs, encoder_hidden):
        """
        decoder_input: [B, rus_seq_len]
        encoder_outputs: [B, eng_seq_len, hidden_size] - ВСЕ скрытые состояния encoder'а!
        encoder_hidden: [1, B, hidden_size] - начальное состояние
        """
        B = decoder_input.size(0)
        
        # Embed decoder input
        embedded = self.embedding(decoder_input)  # [B, rus_seq_len, embed_size]

        if self.pos_encoding != 'none' and self.pe is not None:
            pe = self.pe[:decoder_input.size(1)].unsqueeze(0).to(embedded.device)
            embedded = embedded + pe
        
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

def compute_perplexity_attention(model, encoder, test_loader, rus_char2idx):
    """Правильный Perplexity = exp(-log P(sequence))"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    encoder.eval()
    
    total_nll = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # Encoder outputs
            embedded = encoder.embedding(batch_X)
            encoder_outputs, encoder_hidden = encoder.gru(embedded)
            
            batch_size = batch_X.size(0)
            sos_idx = rus_char2idx['<']
            
            # INFERENCE: генерируем как при переводе!
            decoder_input = torch.full((batch_size, 1), sos_idx, device=device)
            decoder_hidden = encoder_hidden
            
            for t in range(batch_y.size(1) - 1):  # До реальной длины target
                logits, decoder_hidden, _ = model(decoder_input, encoder_outputs, decoder_hidden)
                
                # Log-probability реального токена
                target = batch_y[:, t + 1]  # Сдвиг (игнорируем SOS)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # Последний timestep
                nll_loss = -log_probs.gather(1, target.unsqueeze(1)).squeeze()
                
                # Считаем только непустые токены
                valid_mask = (target != rus_char2idx['.'])
                total_nll += nll_loss[valid_mask].sum().item()
                total_tokens += valid_mask.sum().item()
                
                # Следующий input = предсказанный токен (inference!)
                decoder_input = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
    
    avg_nll = total_nll / total_tokens
    perplexity = np.exp(avg_nll)
    return perplexity


def extract_encoder_outputs(encoder, batch_X):
    """Извлекает ВСЕ скрытые состояния encoder'а"""
    embedded = encoder.embedding(batch_X)
    encoder_outputs, _ = encoder.gru(embedded)  # [B, eng_seq_len, hidden]
    return encoder_outputs

def train_attention_decoder(encoder, train_loader, valid_loader, rus_vocab_size, eng_idx2char, rus_idx2char, 
                          eng_char2idx, rus_char2idx, X_train_t, X_valid_t, max_len, epochs=100, lr=0.0003, patience=15, suffix="", **kwargs):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = False
    
    decoder = AttentionDecoder(rus_vocab_size, hidden_size=encoder.hidden_size, **kwargs).to(device)
    
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
            torch.save(decoder.state_dict(), f'best_attention_decoder_{suffix}.pth')
    
    decoder.load_state_dict(torch.load(f'best_attention_decoder_{suffix}.pth'))
    return decoder, train_losses, valid_losses


def check_translation(X_test_t, eng_char2idx, eng_idx2char, decoder, encoder, rus_char2idx, rus_idx2char, max_len, n):
    translations = []
    for i in range(n):
        eng_indices = [idx.item() for idx in X_test_t[i] if idx.item() not in [eng_char2idx['.'], eng_char2idx['<'], eng_char2idx['>']]]
        eng_name = ''.join([eng_idx2char[idx] for idx in eng_indices])
        rus_pred = decoder.translate(encoder, eng_indices, rus_char2idx, rus_idx2char, max_len)
        translations.append(f"{eng_name:10s}→{rus_pred}")
    print(" | ".join(translations))

def plot_attention_heatmap(eng_name, rus_name, attention_weights_list):

    eng_letters = list(eng_name)
    rus_letters = list(rus_name)
    
    # attention_matrix: [len(rus), len(eng)]
    attention_matrix = np.array(attention_weights_list[:len(rus_letters)])
    
    plt.figure(figsize=(10, len(rus_letters)*0.5))
    sns.heatmap(attention_matrix, annot=True, fmt='.2f', 
                xticklabels=eng_letters, yticklabels=rus_letters,
                cmap='YlOrRd')
    plt.title(f'Attention: "{eng_name}" → "{rus_name}"')
    plt.xlabel('English letters')
    plt.ylabel('Russian letters')
    plt.tight_layout()
    plt.show()

def visualize_attention(encoder, decoder, eng_name_indices, eng_idx2char, rus_char2idx, rus_idx2char, max_len=20):
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
        
        attention_weights_list.append(attn_weights[0].cpu().detach().numpy())  # [eng_len]
        generated_tokens.append(next_token)
        
        if next_token == rus_char2idx['>']:
            break
            
        input_token = torch.tensor([[next_token]], device=device)
        encoder_hidden = hidden
    
    # Русское имя (без SOS)
    rus_name = ''.join([rus_idx2char[idx] for idx in generated_tokens[1:] 
                       if idx != rus_char2idx['.']])
    eng_name = ''.join([eng_idx2char[idx] for idx in eng_name_indices])
    
    return eng_name, rus_name, attention_weights_list

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