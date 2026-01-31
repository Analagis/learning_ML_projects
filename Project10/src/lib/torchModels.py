import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import os
import random

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
    
def set_seed(seed=42):
    """Полная фиксация всего"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # ✅ КРИТИЧНО ДЛЯ DataLoader!
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Основная функция анализа ---
def analyze_vanishing_gradient_torch(X_data, y_data, token2id, n_epochs=3, batch_size=32, hidden_size=64, emb_dim=32):
    """
    Анализ затухающего градиента на стандартных слоях PyTorch (RNN, LSTM, GRU).
    Используем register_full_backward_hook для сбора градиентов embedding слоя.
    """
    print("\n=== Анализ Vanishing Gradient Gender ===")

    set_seed(42)
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
    
    # Для отладки: проверка с batch_size=1 и n_epochs=1
    debug_mode = False
    if batch_size == 1 and n_epochs == 1:
        debug_mode = True
        print("\n=== Debug Mode (batch_size=1, n_epochs=1) ===")
    
    # 2. Цикл по архитектурам
    for rnn_type in ['RNN', 'LSTM', 'GRU']:
        print(f"Training {rnn_type}...")
        
        # Создаем свежую модель
        model = StandardRNNModel(vocab_size, emb_dim, hidden_size, rnn_type).to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        # Хранилище градиентов: (n_epochs, seq_len, batch_size, emb_dim)
        grads_by_epoch = []
        
        # Хук на эмбеддинги для сбора градиентов
        def embedding_backward_hook(module, grad_input, grad_output):
            """
            Хук для сбора градиентов embedding слоя.
            grad_output[0] - градиенты по выходу embedding слоя
            Форма: (batch_size, seq_len, emb_dim)
            """
            if grad_output[0] is not None:
                # Сохраняем градиенты для текущего батча
                # Детализируем по эпохам
                current_epoch_grads = grad_output[0].detach().cpu()
                grads_by_epoch.append(current_epoch_grads)
            
            # Возвращаем None, так как не модифицируем градиенты
            return None
        
        # Регистрируем backward hook на embedding слое
        handle = model.embedding.register_full_backward_hook(embedding_backward_hook)
        
        # Обучение
        for epoch in range(n_epochs):
            epoch_grads = []
            
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                
                optimizer.zero_grad()
                logits = model(bx).view(-1)
                loss = criterion(logits, by)
                loss.backward()
                
                # После backward hook автоматически сработает
                # и добавит градиенты в grads_by_epoch
                
                optimizer.step()
            
            # В отладочном режиме проверяем не-zero градиенты
            if debug_mode and grads_by_epoch:
                print(f"\nDebug info for {rnn_type}:")
                
                # Берем последний батч
                last_grads = grads_by_epoch[-1]  # (1, seq_len, emb_dim) при batch_size=1
                
                # i. Compute the norm of each vector along the last dimension
                # Вычисляем норму по последнему измерению (emb_dim)
                grad_norms = torch.norm(last_grads, dim=-1)  # (1, seq_len)
                
                print(f"Gradient norms shape: {grad_norms.shape}")
                print(f"Gradient norms: {grad_norms.squeeze().numpy()}")
                
                # ii. Compare which tokens have non-zero gradient norm
                # Сравниваем с токенами входной последовательности
                input_tokens = bx.squeeze().cpu().numpy()  # (seq_len,)
                
                print(f"\nInput tokens: {input_tokens}")
                print(f"Non-zero gradient positions:")
                
                for pos in range(grad_norms.shape[1]):
                    norm_val = grad_norms[0, pos].item()
                    token_id = input_tokens[pos]
                    if norm_val > 1e-6:  # порог для "non-zero"
                        print(f"  Position {pos}: token_id={token_id}, norm={norm_val:.6f}")
        
        handle.remove()
        
        results.update(frobenius_norm(grads_by_epoch, loader, n_epochs, batch_size, emb_dim, rnn_type))

    # f. Plot and compare the distribution of mean gradient norm over token position
    plt.figure(figsize=(10, 6))
    for name, vals in results.items():
        limit = len(vals)
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
    
    set_seed(42)
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
        
        def backward_hook(module, grad_input, grad_output):
            """
            grad_output[0] - градиент по выходу RNN (hidden states)
            Форма: (batch_size, seq_len, hidden_size)
            
            grad_input[0] - градиент по входу RNN (embedded input)
            """
            if grad_output[0] is not None:
                # Сохраняем градиенты выходов RNN для анализа
                # grad_output[0].shape = (batch_size, seq_len, hidden_size)
                current_epoch_grads = grad_output[0].detach().cpu()
                grads_storage.append(current_epoch_grads)
            
            # Возвращаем None, так как не модифицируем градиенты
            return None
        
        handle = model.embedding.register_full_backward_hook(backward_hook)
        
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
    
        results.update(frobenius_norm(grads_storage, loader, n_epochs, batch_size, emb_dim, rnn_type))
    
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

def frobenius_norm(grads_storage, loader, n_epochs, batch_size, emb_dim, rnn_type):
    results = {}
    if grads_storage:
        # Преобразуем список в тензор: (n_batches, batch_size, seq_len, emb_dim)
        # Объединяем все батчи
        all_grads = torch.cat(grads_storage, dim=0)  # (total_batches, seq_len, emb_dim)
        
        # Группируем по эпохам (предполагая одинаковое количество батчей на эпоху)
        batches_per_epoch = len(loader)
        n_total_batches = all_grads.shape[0]
        
        # Убедимся, что можем разделить на эпохи
        if n_total_batches >= batches_per_epoch * n_epochs:
            # Решейпим: (n_epochs, batches_per_epoch, batch_size, seq_len, emb_dim)
            all_grads_reshaped = all_grads.view(n_epochs, batches_per_epoch, batch_size, -1, emb_dim)
            
            # Вычисляем норму Фробениуса для каждой позиции токена в каждой эпохе
            # Норма Фробениуса = L2 норма для матрицы (batch_size × emb_dim)
            # Для каждого (epoch, token_position) получаем скаляр
            epoch_token_norms = []
            
            for epoch_idx in range(n_epochs):
                epoch_grads = all_grads_reshaped[epoch_idx]  # (batches_per_epoch, batch_size, seq_len, emb_dim)
                
                # Объединяем все батчи эпохи
                epoch_all_grads = epoch_grads.reshape(-1, epoch_grads.shape[-2], emb_dim)
                
                # Для каждой позиции токена вычисляем норму Фробениуса
                seq_len = epoch_all_grads.shape[1]
                token_norms = []
                
                for token_pos in range(seq_len):
                    # Градиенты для данной позиции токена
                    token_grads = epoch_all_grads[:, token_pos, :]  # (n_samples, emb_dim)
                    
                    # Норма Фробениуса: sqrt(sum of squares of all elements)
                    fro_norm = torch.norm(token_grads, p='fro').item()
                    token_norms.append(fro_norm)
                
                epoch_token_norms.append(token_norms)
            
            # e. Берем среднее по эпохам
            epoch_token_norms_tensor = torch.tensor(epoch_token_norms)  # (n_epochs, seq_len)
            mean_norms = epoch_token_norms_tensor.mean(dim=0).numpy()  # (seq_len,)
            
            results[rnn_type] = mean_norms
        else:
            # Упрощенный расчет, если не получается разделить по эпохам
            print(f"Warning: Cannot split gradients by epochs for {rnn_type}")
            print(f"Total batches: {n_total_batches}, Expected: {batches_per_epoch * n_epochs}")
            
            # Альтернатива: считаем среднее по всем батчам
            all_grads_reshaped = all_grads  # (total_batches, seq_len, emb_dim)
            
            # Вычисляем норму Фробениуса для каждой позиции токена
            seq_len = all_grads_reshaped.shape[1]
            token_norms = []
            
            for token_pos in range(seq_len):
                token_grads = all_grads_reshaped[:, token_pos, :]  # (total_batches, emb_dim)
                fro_norm = torch.norm(token_grads, p='fro').item()
                token_norms.append(fro_norm)
            
            results[rnn_type] = np.array(token_norms)
    else:
        print(f"Warning: No grads for {rnn_type}")

    return results