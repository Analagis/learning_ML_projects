import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from baseVAE import BaseVAE

class SupervisedVAE(BaseVAE):
    """
    Supervised Variational AutoEncoder.
    Добавляет классификацию цифр (0-9) поверх латентного пространства z.
    """
    
    def __init__(self, input_dim: int, n_classes: int = 10, hidden_dims: list = None, 
                 z_dim: int = 2, beta: float = 0.1, classifier_hidden: list = [64]):
        super().__init__(input_dim, hidden_dims, z_dim, beta)
        
        self.n_classes = n_classes
        self.classifier = self._build_classifier(classifier_hidden)
    
    def _build_classifier(self, hidden_dims: list) -> nn.Module:
        """Классификатор: z → класс."""
        dims = [self.z_dim] + hidden_dims + [self.n_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
            if i < len(dims) - 2:  # BatchNorm везде кроме последнего
                layers.append(nn.BatchNorm1d(dims[i+1]))
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Перехватываем encode для доступа к z."""
        mu, logvar = super().encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z  # Возвращаем z тоже!
    
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> dict:
        """
        Полный supervised forward.
        Returns: dict с x_recon, mu, logvar, z, y_pred (если y есть).
        """
        mu, logvar, z = self.encode(x)
        x_recon = self.decode(z)
        y_pred = self.classifier(z)
        
        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'y_pred': y_pred
        }
    
    def compute_supervised_loss(self, x: torch.Tensor, y: torch.Tensor, 
                              x_recon: torch.Tensor, mu: torch.Tensor, 
                              logvar: torch.Tensor, y_pred: torch.Tensor) -> tuple:
        """
        Supervised ELBO + CrossEntropy.
        """
        # Генеративная часть (ELBO)
        elbo_loss, elbo_metrics = super().compute_elbo_loss(x, x_recon, mu, logvar)
        
        # Supervised часть
        cls_loss = F.cross_entropy(y_pred, y)
        
        # Итоговая loss
        total_loss = elbo_loss + cls_loss  # Можно добавить вес: + lambda_cls * cls_loss
        
        metrics = {**elbo_metrics, 'cls_loss': cls_loss.item(), 'total_loss': total_loss.item()}
        return total_loss, metrics
    
    def fit(self, train_loader, val_loader=None, epochs: int = 50, lr: float = 1e-3,
            device: str = None, log_every: int = 100, save_path: str = None) -> list:
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.to(device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                
                # ✅ ПРОСТОЙ forward
                outputs = self(x)
                
                # ✅ ПРЯМОЕ распаковывание (без фильтрации!)
                loss, metrics = self.compute_supervised_loss(
                    x, y, outputs['x_recon'], outputs['mu'], 
                    outputs['logvar'], outputs['y_pred']
                )
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % log_every == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'recon': f'{metrics["recon_loss"]:.4f}',
                        'kl': f'{metrics["kl_loss"]:.4f}',
                        'cls': f'{metrics["cls_loss"]:.4f}'
                    })
            
            avg_loss = train_loss / num_batches
            losses.append(avg_loss)
            print(f'Epoch {epoch+1:2d}: Loss={avg_loss:.4f}')
        
        return losses
    
    def _supervised_validate(self, val_loader, device):
        """Валидация с accuracy."""
        self.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                loss, metrics = self.compute_supervised_loss(x, y, **outputs)
                total_loss += metrics['total_loss']
                
                pred = outputs['y_pred'].argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        self.train()
        return total_loss / len(val_loader), correct / total
    
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Предсказание класса для новых данных."""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            return outputs['y_pred'].argmax(dim=1)
