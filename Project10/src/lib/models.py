import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentModel(nn.Module):
    """
    Общая модель:
    - Embedding
    - RNN-блок (VanillaRNNBlock / GRUBlock / LSTMBlock)
    - Голова для следующего токена
    - Голова для пола
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        rnn_block_cls,      # класс блока, напр. VanillaRNNBlock
        dropout_p=0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 1. Эмбеддинги
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Рекуррентный блок
        self.rnn = rnn_block_cls(input_size=embedding_dim, hidden_size=hidden_size)

        # 3. Dropout (опционально)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # 4. Линейный слой для предсказания следующего токена
        self.token_head = nn.Linear(hidden_size, vocab_size)

        # 5. Линейный слой для логита пола
        self.gender_head = nn.Linear(hidden_size, 1)

        # 6. Сигмоида для вероятности пола
        self.gender_activation = nn.Sigmoid()

    def init_hidden(self, batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Инициализация скрытого состояния для RNN.
        Для vanilla RNN и GRU достаточно одного тензора (batch, hidden_size).
        Для LSTM здесь потом можно будет вернуть кортеж (h0, c0).
        """
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) — индексы токенов
        hidden: начальное скрытое состояние (или None)

        Возвращает:
        - token_logits: (batch, seq_len, vocab_size)
        - gender_logit: (batch, 1)
        - gender_prob:  (batch, 1)
        - last_hidden:  (batch, hidden_size) — пригодится для генерации/анализа
        """
        # 1. Индексы -> эмбеддинги
        emb = self.embedding(x)                 # (batch, seq_len, embedding_dim)
        emb = self.dropout(emb)

        # 2. Прогон через RNN-блок
        outputs, last_hidden = self.rnn(emb, hidden)  # outputs: (batch, seq_len, hidden_size)

        outputs = self.dropout(outputs)

        # 3. Логиты токенов для каждого шага
        token_logits = self.token_head(outputs)       # (batch, seq_len, vocab_size)

        # 4. Логит пола по последнему скрытому состоянию
        gender_logit = self.gender_head(last_hidden)  # (batch, 1)
        gender_prob = self.gender_activation(gender_logit)

        return token_logits, gender_logit, gender_prob, last_hidden

    @torch.no_grad()
    def generate_name(
        self,
        token2id,
        id2token,
        start_text="",
        max_len=20,
        temperature=1.0,
        sos_token="<",
        eos_token=">",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Стохастическая генерация имени:
        - сэмплирование следующего токена по softmax(logits / temperature)
        - можно задать начало имени (start_text)
        - temperature управляет степенью случайности
        """
        self.eval()

        # 1. начальная последовательность токенов
        if start_text:
            init_tokens = list(start_text.lower())
        else:
            init_tokens = [sos_token]

        # проверка, что все есть в словаре
        for ch in init_tokens:
            if ch not in token2id:
                raise ValueError(f"Символ '{ch}' отсутствует в словаре токенов")

        # 2. индексы и прогрев скрытого состояния
        input_ids = [token2id[ch] for ch in init_tokens]
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=device
        ).unsqueeze(0)  # (1, seq_len)

        hidden = self.init_hidden(batch_size=1, device=device)
        token_logits, _, _, hidden = self(input_tensor, hidden=hidden)

        generated_tokens = init_tokens.copy()

        # 3. пошаговая генерация
        for _ in range(max_len):
            last_token_id = token2id[generated_tokens[-1]]
            last_input = torch.tensor([[last_token_id]], dtype=torch.long, device=device)  # (1, 1)

            token_logits, _, _, hidden = self(last_input, hidden=hidden)
            logits = token_logits[:, -1, :].squeeze(0)          # (vocab_size,)

            # temperature + softmax
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)                   # (vocab_size,)

            next_id = torch.multinomial(probs, num_samples=1).item()
            next_token = id2token[next_id]

            if next_token == eos_token:
                break

            generated_tokens.append(next_token)

        # 4. убираем служебные токены
        name_chars = [ch for ch in generated_tokens if ch not in (sos_token, eos_token)]
        name = "".join(name_chars)

        return name
    
    @torch.no_grad()
    def predict_gender(self, x, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        x: тензор индексов токенов (batch, seq_len)

        Возвращает:
        - probs: тензор (batch, 1) — вероятность "класс 1" (например, girl)
        - logits: тензор (batch, 1) — сырые логиты
        """
        self.eval()
        x = x.to(device)

        with torch.no_grad():
            # token_logits нам тут не нужны
            _, gender_logit, gender_prob, _ = self(x)

        return gender_prob, gender_logit

class VanillaRNNBlock(nn.Module):
    """
    Простой RNN-блок: на вход (embeddings, h0) -> на выход (outputs, h_last)
    embeddings: (batch, seq_len, input_size)
    h0: (batch, hidden_size) или None
    outputs: (batch, seq_len, hidden_size)
    h_last: (batch, hidden_size)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Параметры W_x, W_h, b задаём явно (не через nn.RNN)
        self.W_x = nn.Linear(input_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h0=None):
        """
        x: (batch, seq_len, input_size)
        h0: (batch, hidden_size) или None
        """
        batch_size, seq_len, _ = x.size()

        if h0 is None:
            h_t = x.new_zeros(batch_size, self.hidden_size)
        else:
            h_t = h0

        outputs = []

        # разворот по времени
        for t in range(seq_len):
            x_t = x[:, t, :]                    # (batch, input_size)
            h_t = torch.tanh(self.W_x(x_t) + self.W_h(h_t))  # (batch, hidden_size)
            outputs.append(h_t.unsqueeze(1))    # (batch, 1, hidden_size)

        outputs = torch.cat(outputs, dim=1)     # (batch, seq_len, hidden_size)
        h_last = h_t                            # (batch, hidden_size)

        return outputs, h_last
