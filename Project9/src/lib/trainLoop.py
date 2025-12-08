import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=1e-3,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device).float().view(-1, 1)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).float().view(-1, 1)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                all_targets.extend(targets.cpu().numpy().ravel())
                all_probs.extend(probs.cpu().numpy().ravel())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        roc_auc = roc_auc_score(all_targets, all_probs)
        return epoch_loss, roc_auc

    def fit(self, num_epochs):
        best_roc_auc = 0.0
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_roc_auc = self.validate()

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_ROC_AUC: {val_roc_auc:.4f}"
            )

            if val_roc_auc > best_roc_auc:
                best_roc_auc = val_roc_auc

        print(f"Best val ROC AUC: {best_roc_auc:.4f}")
