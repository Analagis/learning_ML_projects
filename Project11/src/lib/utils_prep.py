from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from collections import Counter

def build_vocab(texts, specials):
    """
    names_processed: итерируемый объект строк
        (например, df["name_processed"])
    
    Возвращает:
        char2idx    - dict: символ -> индекс
        idx2char    - dict: индекс -> символ
    """
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    
    unique_chars = specials + sorted(list(all_chars - set(specials)))
    
    char2idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    
    return char2idx, idx2char

def encode_name(name, char2idx, SOS_token, EOS_token):
    """name уже в нижнем регистре!"""
    indices = [char2idx[SOS_token]]
    for char in name:
        if char in char2idx:
            indices.append(char2idx[char])
    indices.append(char2idx[EOS_token])
    return indices

def pad_sequences(sequences, pad_idx, max_length=None):
    """
    Паддит все последовательности до max_length (или до самой длинной)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    
    for seq in sequences:
        # Добавляем PAD справа до нужной длины
        padded_seq = seq + [pad_idx] * (max_length - len(seq))
        padded.append(padded_seq)
    
    return padded, max_length
