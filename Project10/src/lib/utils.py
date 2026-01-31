from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from collections import Counter

def build_vocab(names_processed):
    """
    names_processed: итерируемый объект строк
        (например, df["name_processed"])
    
    Возвращает:
        tokens      - отсортированный список уникальных символов
        token2id    - dict: символ -> индекс
        id2token    - dict: индекс -> символ
    """
    # 1. собираем множество всех символов из всех имён
    all_chars = set()
    for name in names_processed:
        all_chars.update(name)
    # 2. делаем детерминированный порядок (например, сортировка)
    tokens = sorted(all_chars)
    
    # 3. строим отображения
    token2id = {ch: i for i, ch in enumerate(tokens)}
    id2token = {i: ch for ch, i in token2id.items()}
    
    return tokens, token2id, id2token


def compute_roc_auc_splits(model, train_loader, valid_loader, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()

    def collect_probs_and_targets(loader):
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).float().view(-1)

                # используем predict_gender
                probs, _ = model.predict_gender(x_batch, device=device)  # (batch, 1)
                probs = probs.view(-1)  # (batch,)

                all_probs.append(probs.cpu())
                all_targets.append(y_batch.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        return all_targets, all_probs

    y_train, p_train = collect_probs_and_targets(train_loader)
    y_valid, p_valid = collect_probs_and_targets(valid_loader)
    y_test,  p_test  = collect_probs_and_targets(test_loader)

    roc_train = roc_auc_score(y_train, p_train)
    roc_valid = roc_auc_score(y_valid, p_valid)
    roc_test  = roc_auc_score(y_test,  p_test)

    return roc_train, roc_valid, roc_test

def calculate_perplexity(logits, targets, ignore_index=-100):
    """
    Расчет Perplexity двумя способами:
    1. ppl_ce: Через torch.nn.functional.cross_entropy (стандартный, стабильный).
    2. ppl_manual: Вручную через вероятности (softmax -> gather -> log -> exp).
    
    Args:
        logits (torch.Tensor): Логиты модели, форма (Batch, Seq_Len, Vocab_Size) или (N, Vocab_Size)
        targets (torch.Tensor): Истинные индексы токенов, форма (Batch, Seq_Len) или (N,)
        ignore_index (int): Индекс токена (паддинга), который игнорируется при расчете.
        
    Returns:
        dict: {'ppl_ce': float, 'ppl_manual': float}
    """
    # 1. Приводим к плоскому виду: (N, Vocab_Size) и (N,)
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if targets.dim() == 2:
        targets = targets.view(-1)
        
    # --- Способ 1: Через CrossEntropyLoss ---
    # CrossEntropyLoss уже содержит LogSoftmax + NLLLoss
    ce_loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='mean')
    ppl_ce = torch.exp(ce_loss).item()
    
    # --- Способ 2: Вручную (Educational) ---
    # a) Считаем вероятности для каждого слова в словаре
    probs = F.softmax(logits, dim=-1)  # (N, Vocab)
    
    # b) Выбираем вероятности, соответствующие истинным таргетам
    # Нам нужны только те позиции, где target != ignore_index
    mask = targets != ignore_index
    valid_targets = targets[mask]
    valid_probs_matrix = probs[mask]
    
    # gather берет из каждой строки вероятность правильного класса
    # (N_valid, 1)
    target_probs = valid_probs_matrix.gather(1, valid_targets.unsqueeze(1)).squeeze()
    
    # c) Negative Log Likelihood
    # Добавляем epsilon для стабильности, чтобы log(0) не дал -inf
    eps = 1e-9
    nll = -torch.log(target_probs + eps)
    
    # d) Усредняем и берем экспоненту
    mean_nll = nll.mean()
    ppl_manual = torch.exp(mean_nll).item()
    
    return {'ppl_ce': ppl_ce, 'ppl_manual': ppl_manual}

def evaluate_baselines(test_loader, vocab_size, token2id, device='cuda'):
    all_targets = []
    char_counter = Counter()
    total_chars = 0
    pad_idx = token2id.get('<PAD>', 0)

    for x_batch, _ in test_loader:
        targets = x_batch[:, 1:].to(device)
        all_targets.append(targets)
        flat = targets.flatten().tolist()
        non_pad = [c for c in flat if c != pad_idx]
        char_counter.update(non_pad)
        total_chars += len(non_pad)

    full_targets = torch.cat(all_targets, dim=0).view(-1)
    
    # Случайная модель
    print(f"Random Model PPL: {vocab_size:.4f}")

    # Частотная модель
    freq_probs = torch.zeros(vocab_size).to(device)
    for tid, count in char_counter.items():
        if tid < vocab_size: freq_probs[tid] = count / total_chars
    
    freq_logits = torch.log(freq_probs + 1e-9).unsqueeze(0).expand(len(full_targets), -1)
    
    loss = torch.nn.functional.cross_entropy(freq_logits, full_targets, ignore_index=pad_idx)
    freq_ppl = torch.exp(loss).item()
    
    print(f"Frequency Model PPL: {freq_ppl:.3f}")