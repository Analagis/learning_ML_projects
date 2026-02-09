from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from collections import Counter
import torch.nn as nn
import numpy as np
import time
from functools import wraps

def check_translation(X_test_t, eng_char2idx, eng_idx2char, decoder, encoder, rus_char2idx, rus_idx2char, y_max_len, n):
    translations = []
    for i in range(n):
        eng_indices = [idx.item() for idx in X_test_t[i] if idx.item() not in [eng_char2idx['.'], eng_char2idx['<'], eng_char2idx['>']]]
        eng_name = ''.join([eng_idx2char[idx] for idx in eng_indices])
        rus_pred = decoder.translate(encoder, eng_indices, rus_char2idx, rus_idx2char, y_max_len)
        translations.append(f"{eng_name:10s}→{rus_pred}")
    print(" | ".join(translations))

def compute_perplexity(model, encoder, test_loader, rus_char2idx, teacher_forcing=False, attention=False):
    """Perplexity с Teacher Forcing или Inference"""
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
            
            # Encoder outputs
            if attention:
                encoder_outputs = encoder.get_encoder_outputs(batch_X)
            encoder_hidden = encoder.get_encoder_state(batch_X)
            
            batch_size = batch_X.size(0)
            sos_idx = rus_char2idx['<']
            input_token = torch.full((batch_size, 1), sos_idx, device=device)
            decoder_hidden = encoder_hidden
            
            for t in range(1, batch_y.size(1)):
                if attention:
                    logits, decoder_hidden, _ = model(input_token, encoder_outputs, decoder_hidden)
                else:
                    logits, decoder_hidden = model(input_token, decoder_hidden)
                target = batch_y[:, t]
                
                loss = criterion(logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1))
                total_loss += loss.item()
                total_tokens += (target != rus_char2idx['.']).sum().item()
                
                # Выбор режима
                if teacher_forcing:
                    input_token = batch_y[:, t:t+1]  # Teacher forcing
                else:
                    input_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # Inference
    
    avg_nll = total_loss / total_tokens
    perplexity = np.exp(avg_nll)
    return perplexity

def timer(func):
    @wraps(func)
    def wrapper(*args, epochs=10, **kwargs):
        start_time = time.time()
        result = func(*args, epochs=epochs, **kwargs)
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        mins, secs = divmod(remainder, 60)
        print(f"\033[94mВремя обучения {epochs} эпох: {hours:02d}:{mins:02d}:{secs:02d}\033[0m")
        return result
    return wrapper