import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


class NameTrainer:
    def __init__(
        self,
        model,
        optimizer=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_next_char_loss=True,
        use_gender_loss=False,
        alpha=1.0,   # вес для next-char
        beta=1.0     # вес для gender
    ):
        self.model = model.to(device)
        self.optimizer = optimizer if optimizer else optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        self.device = device

        self.use_next_char_loss = use_next_char_loss
        self.use_gender_loss = use_gender_loss
        self.alpha = alpha
        self.beta = beta

    def _compute_losses(self, x, y_gender):
        """
        x: (batch, seq_len) индексы токенов
        y_gender: (batch,) метки пола 0/1 или None
        """
        x = x.to(self.device)
        if y_gender is not None:
            y_gender = y_gender.to(self.device).float().view(-1, 1)

        # прямой проход
        token_logits, gender_logit, gender_prob, _ = self.model(x)

        total_loss = 0.0
        loss_next = None
        loss_gender = None

        # ----- loss для следующей буквы -----
        if self.use_next_char_loss:
            # logits для всех позиций, кроме последней
            logits = token_logits[:, :-1, :]   # (batch, seq_len-1, vocab_size)
            # таргеты — "сдвинутые" токены
            targets = x[:, 1:]                 # (batch, seq_len-1)

            # приводим к (N, C) и (N,)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss_next = F.cross_entropy(logits, targets)
            total_loss = total_loss + self.alpha * loss_next

        # ----- loss для пола -----
        if self.use_gender_loss and y_gender is not None:
            # gender_logit: (batch, 1)
            # используем BCEWithLogitsLoss, поэтому лучше логиты, а не сигмоиду
            loss_gender = F.binary_cross_entropy_with_logits(
                gender_logit, y_gender
            )
            total_loss = total_loss + self.beta * loss_gender

        return total_loss, loss_next, loss_gender

    def _step_batch(self, batch, train=True):
        # ожидаем batch = (x, y_gender) или просто x
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y_gender = batch
        else:
            x, y_gender = batch, None

        self.model.train(mode=train)

        if train:
            self.optimizer.zero_grad()

        loss, loss_next, loss_gender = self._compute_losses(x, y_gender)

        if train:
            loss.backward()
            self.optimizer.step()

        # возвращаем скаляры (могут быть None)
        loss_value = float(loss.item())
        loss_next_value = float(loss_next.item()) if loss_next is not None else None
        loss_gender_value = float(loss_gender.item()) if loss_gender is not None else None

        return loss_value, loss_next_value, loss_gender_value

    def train_epoch(self, train_loader: DataLoader):
        total_loss = 0.0
        total_next = 0.0
        total_gender = 0.0
        n_batches = 0
        n_next = 0
        n_gender = 0

        for batch in train_loader:
            loss, loss_next, loss_gender = self._step_batch(batch, train=True)
            n_batches += 1
            total_loss += loss

            if loss_next is not None:
                total_next += loss_next
                n_next += 1

            if loss_gender is not None:
                total_gender += loss_gender
                n_gender += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_next = total_next / max(1, n_next) if n_next > 0 else None
        avg_gender = total_gender / max(1, n_gender) if n_gender > 0 else None

        return avg_loss, avg_next, avg_gender

    def eval_epoch(self, valid_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_next = 0.0
        total_gender = 0.0
        n_batches = 0
        n_next = 0
        n_gender = 0

        with torch.no_grad():
            for batch in valid_loader:
                loss, loss_next, loss_gender = self._step_batch(batch, train=False)
                n_batches += 1
                total_loss += loss

                if loss_next is not None:
                    total_next += loss_next
                    n_next += 1

                if loss_gender is not None:
                    total_gender += loss_gender
                    n_gender += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_next = total_next / max(1, n_next) if n_next > 0 else None
        avg_gender = total_gender / max(1, n_gender) if n_gender > 0 else None

        return avg_loss, avg_next, avg_gender
