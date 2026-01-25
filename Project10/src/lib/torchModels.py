import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

# --- Класс-обертка для стандартных слоев ---
class StandardRNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, rnn_type='RNN'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # Используем нативные слои PyTorch
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, 1) # Для классификации пола
        self.rnn_type = rnn_type

    def forward(self, x):
        # x: (Batch, Seq)
        emb = self.embedding(x) # (Batch, Seq, Emb)
        
        # Стандартные RNN возвращают (output, hidden)
        out, hidden = self.rnn(emb)
        
        # Нам нужен последний скрытый стейт для классификации
        if self.rnn_type == 'LSTM':
            # hidden это (h_n, c_n), берем h_n
            h_last = hidden[0] 
        else:
            h_last = hidden
            
        # h_last shape: (Num_Layers, Batch, Hidden) -> берем последний слой
        h_last = h_last[-1] 
        
        logits = self.fc(h_last) # (Batch, 1)
        return logits
    
class StandardRNNGen(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, rnn_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if rnn_type == 'RNN': self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM': self.rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        elif rnn_type == 'GRU': self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size) # Предсказание следующей буквы

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb) # out: (Batch, Seq, Hidden) - последовательность!
        logits = self.fc(out)  # (Batch, Seq, Vocab)
        return logits

# --- Основная функция анализа ---
def analyze_vanishing_gradient_torch(X_data, y_data, token2id, n_epochs=3, batch_size=32, hidden_size=64, emb_dim=32):
    """
    Анализ затухающего градиента на стандартных слоях PyTorch (RNN, LSTM, GRU).
    """
    print("\n=== Анализ Vanishing Gradient (Standard PyTorch Models) Gender ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Подготовка данных (Фильтрация по моде длины)
    if not torch.is_tensor(X_data): X_data = torch.tensor(X_data, dtype=torch.long)
    if not torch.is_tensor(y_data): 
        vals = y_data.values if hasattr(y_data, 'values') else y_data
        y_data = torch.tensor(vals).float()
    else:
        y_data = y_data.float()

    pad_idx = token2id.get('<PAD>', 0)
    bos_idx = token2id.get('<', -1)
    eos_idx = token2id.get('>', -1)
    
    # Считаем длину (исключая спецсимволы)
    mask = (X_data != pad_idx)
    if bos_idx != -1: mask &= (X_data != bos_idx)
    if eos_idx != -1: mask &= (X_data != eos_idx)
    lengths = mask.sum(dim=1).numpy()
    
    mode_len = int(Counter(lengths).most_common(1)[0][0])
    print(f"Mode length: {mode_len}")
    
    indices = np.where(lengths == mode_len)[0]
    # Берем подвыборку (можно ограничить 1000 примеров для скорости)
    if len(indices) > 2000: indices = indices[:2000]
        
    dataset = TensorDataset(X_data[indices], y_data[indices])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    vocab_size = len(token2id)
    results = {}
    
    # 2. Цикл по архитектурам
    for rnn_type in ['RNN', 'LSTM', 'GRU']:
        print(f"Training {rnn_type}...")
        
        # Создаем свежую модель
        model = StandardRNNModel(vocab_size, emb_dim, hidden_size, rnn_type).to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        grads_storage = []
        
        # Хук на эмбеддинги
        def forward_hook(module, inp, out):
            if out.requires_grad:
                out.register_hook(lambda g: grads_storage.append(g.detach().cpu()))
        
        handle = model.embedding.register_forward_hook(forward_hook)
        
        # Обучение
        for epoch in range(n_epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                
                optimizer.zero_grad()
                logits = model(bx).view(-1)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        
        handle.remove()
        
        # Анализ градиентов
        if grads_storage:
            full_grads = torch.cat(grads_storage, dim=0) # (Total_N, Seq, Emb)
            # Считаем норму
            norms = torch.norm(full_grads, p='fro', dim=-1) # (Total_N, Seq)
            mean_norms = norms.mean(dim=0).numpy() # (Seq,)
            results[rnn_type] = mean_norms
        else:
            print(f"Warning: No grads for {rnn_type}")

    # 3. Визуализация
    plt.figure(figsize=(10, 6))
    for name, vals in results.items():
        limit = min(len(vals), mode_len + 3)
        plt.plot(range(limit), vals[:limit], marker='o', label=name, linewidth=2)
        
    plt.title("Vanishing Gradient Analysis (Standard PyTorch Models)")
    plt.xlabel("Token Position (Input Sequence)")
    plt.ylabel("Gradient Norm (Influence on Output)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_vanishing_gradient_torch_name(X_data, token2id, n_epochs=3, batch_size=32, hidden_size=64, emb_dim=32):
    """
    Анализ градиентов для задачи генерации имен (Next Token Prediction).
    """
    print("\n=== Анализ Vanishing Gradient (Task: Name Generation) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(token2id)
    
    # 1. Подготовка данных
    if not torch.is_tensor(X_data): X_data = torch.tensor(X_data, dtype=torch.long)
    
    # Фильтрация по моде (аналогично)
    pad_idx = token2id.get('<PAD>', 0)
    bos_idx = token2id.get('<', -1)
    eos_idx = token2id.get('>', -1)
    
    mask = (X_data != pad_idx)
    if bos_idx != -1: mask &= (X_data != bos_idx)
    if eos_idx != -1: mask &= (X_data != eos_idx)
    lengths = mask.sum(dim=1).numpy()
    
    mode_len = int(Counter(lengths).most_common(1)[0][0])
    print(f"Mode length: {mode_len}")
    
    indices = np.where(lengths == mode_len)[0]
    if len(indices) > 2000: indices = indices[:2000]
    
    # Dataset содержит только X, так как y мы делаем сдвигом
    dataset = TensorDataset(X_data[indices])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    results = {}
    
    for rnn_type in ['RNN', 'LSTM', 'GRU']:
        print(f"Training {rnn_type}...")
        model = StandardRNNGen(vocab_size, emb_dim, hidden_size, rnn_type).to(device)
        model.train()
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        grads_storage = []
        
        def forward_hook(module, inp, out):
            if out.requires_grad:
                out.register_hook(lambda g: grads_storage.append(g.detach().cpu()))
        
        handle = model.embedding.register_forward_hook(forward_hook)
        
        for epoch in range(n_epochs):
            for batch in loader:
                bx = batch[0].to(device) # (Batch, Seq)
                
                opt.zero_grad()
                
                # Задача: предсказать след. токен
                # Вход: символы 0..N-1
                # Цель: символы 1..N
                inp = bx[:, :-1]
                tgt = bx[:, 1:]
                
                logits = model(inp) # (Batch, Seq-1, Vocab)
                
                # Reshape для CrossEntropy
                loss = crit(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                
                loss.backward()
                opt.step()
        
        handle.remove()
        
        if grads_storage:
            full = torch.cat(grads_storage, dim=0) # (Total_N, Seq-1, Emb)
            norms = torch.norm(full, p='fro', dim=-1).mean(dim=0).numpy()
            results[rnn_type] = norms
    
    # 4. Визуализация
    plt.figure(figsize=(10, 6))
    for name, vals in results.items():
        plt.plot(vals, marker='o', label=name, linewidth=2)
            
    plt.title("Gradient Norm (Name Generation Task)")
    plt.xlabel("Token Position")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()