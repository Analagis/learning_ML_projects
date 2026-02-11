import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class BaseVAE(nn.Module):
    """
    Базовый Variational AutoEncoder с ELBO loss.
    
    Args:
        input_dim (int): Размерность входных данных (например, 784 для MNIST).
        hidden_dims (list): Слои скрытых сетей encoder/decoder.
        z_dim (int): Размерность латентного пространства (по умолчанию 2).
        beta (float): Веса KL терма в ELBO (по умолчанию 1.0).
    """
    
    def __init__(self, input_dim: int, hidden_dims: list|None = None, z_dim: int = 2, beta: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = beta
        
        # Стандартная архитектура encoder/decoder
        self.encoder = self._build_encoder(hidden_dims)
        self.decoder = self._build_decoder(hidden_dims)
        
        # Параметры латентного распределения
        self.fc_mu = nn.Linear(hidden_dims[-1] if hidden_dims else 400, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] if hidden_dims else 400, z_dim)
    
    def _build_encoder(self, hidden_dims: list) -> nn.Module:
        """Строит encoder: x → hidden → mu, logvar."""
        dims = [self.input_dim] + (hidden_dims or [400, 200])
        layers = []
        for h_in, h_out in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(h_in, h_out), nn.ReLU(), nn.BatchNorm1d(h_out)])
        return nn.Sequential(*layers)
    
    def _build_decoder(self, hidden_dims: list) -> nn.Module:
        """Строит decoder: z → hidden → x."""
        dims = [self.z_dim] + (hidden_dims or [400, 200])[::-1] + [self.input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
            if i < len(dims) - 2:  # BatchNorm везде кроме последнего
                layers.append(nn.BatchNorm1d(dims[i+1]))
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder: возвращает mu и logvar."""
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sampling z ~ N(mu, std) с градиентами."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder: z → reconstructed x."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Полный forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def compute_elbo_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                         mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        ELBO loss = Reconstruction + KL divergence.
        
        Returns: (total_loss, loss_dict)
        """
        # Reconstruction loss (BCE для бинарных изображений, MSE для RGB)
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon.view(-1, self.input_dim), 
            x.view(-1, self.input_dim), 
            reduction='mean'
        )
        
        # KL divergence: D_KL(q(z|x) || p(z)) ≈ 0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        elbo_loss = recon_loss + self.beta * kl_loss
        
        return elbo_loss, {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'elbo_loss': elbo_loss.item()
        }
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Генерация новых сэмплов из prior N(0,1)."""
        z = torch.randn(n_samples, self.z_dim).to(device)
        with torch.no_grad():
            samples = self.decode(z).view(n_samples, *self.input_shape)
        return samples
    
    @property
    def input_shape(self):
        """Размер входного изображения"""
        return (1, 28, 28)  # По умолчанию MNIST

    def fit(self, 
            train_loader, 
            epochs: int = 50, 
            lr: float = 1e-3,
            device: str = None,
            log_every: int = 100,
            save_path: str = None) -> list:
        """
        Обучение VAE с валидацией и логированием.
        
        Returns:
            list: Список средних loss по эпохам.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.to(device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0
            
            # Training loop
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
            for batch_idx, (x, _) in enumerate(pbar):
                x = x.to(device)
                
                # Forward
                x_recon, mu, logvar, z = self(x)
                loss, metrics = self.compute_elbo_loss(x, x_recon, mu, logvar)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # Логирование
                if batch_idx % log_every == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'recon': f'{metrics["recon_loss"]:.4f}',
                        'kl': f'{metrics["kl_loss"]:.4f}'
                    })
            
            avg_loss = train_loss / num_batches
            losses.append(avg_loss)

            if save_path and (epoch == 0 or avg_loss < min(losses[:-1])):
                torch.save(self.state_dict(), save_path)
        
        return losses
    
    def plot_latent_space(self, test_loader, n_samples: int = 5000, figsize: tuple = (10, 8)):
        """
        Универсальная визуализация латентного пространства.
        Работает как с BaseVAE (tuple), так и с SVAE (dict).
        """
        import matplotlib.pyplot as plt
        
        self.eval()
        z_list, labels_list = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(z_list) >= n_samples:
                    break
                
                # ✅ Универсальная обработка batch
                if len(batch) == 2:  # (x, labels)
                    x, labels = batch
                else:  # Только x
                    x, labels = batch, None
                
                x = x.to(next(self.parameters()).device)
                
                # ✅ Универсальный forward
                output = self(x)
                
                # ✅ Извлекаем z универсально
                if isinstance(output, dict):
                    z = output['z']  # SVAE
                elif isinstance(output, tuple) and len(output) >= 4:
                    z = output[3]    # BaseVAE: x_recon, mu, logvar, z
                else:
                    raise ValueError("Неизвестный формат вывода модели")
                
                z_list.append(z.cpu())
                if labels is not None:
                    labels_list.append(labels)
        
        # Собираем результаты
        z_all = torch.cat(z_list)[:n_samples]
        
        # Цвета по лейблам ИЛИ случайные
        if labels_list:
            labels_all = torch.cat(labels_list)[:n_samples]
            colors = labels_all.numpy()
        else:
            colors = np.random.randint(0, 10, size=z_all.shape[0])
        
        # Plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=colors, cmap='tab10', alpha=0.7, s=10)
        plt.colorbar(scatter, label='Digit' if labels_list else 'Random')
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.title('VAE Latent Space')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_latent_grid(self, grid_size: int = 15, latent_range: float = 3.0, figsize: int = 15):
        """
        Создает сетку 15x15 изображений из латентного пространства.
        
        Args:
            grid_size: Размер сетки (15x15)
            latent_range: Диапазон z [-range, +range]
            figsize: Размер фигуры
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        self.eval()
        
        # 1. Создаем равномерную сетку в 2D латентном пространстве
        x_linspace = np.linspace(-latent_range, latent_range, grid_size)
        z_grid = np.meshgrid(x_linspace, x_linspace)  # 15x15 координаты
        z_grid = np.stack(z_grid, axis=-1).reshape(-1, 2)  # [225, 2]
        
        # 2. Прогоняем через decoder
        with torch.no_grad():
            z_tensor = torch.tensor(z_grid, dtype=torch.float32).to(next(self.parameters()).device)
            generated_images = self.decode(z_tensor).cpu()  # [225, 784]
        
        # 3. Reshape в картинки 28x28
        generated_images = generated_images.view(-1, 1, 28, 28)  # [225, 1, 28, 28]
        
        # 4. Создаем большую картинку 15x15
        figure = np.zeros((grid_size * 28, grid_size * 28))
        for i, img in enumerate(generated_images):
            row = i // grid_size
            col = i % grid_size
            figure[row*28:(row+1)*28, col*28:(col+1)*28] = img[0]  # Grayscale
        
        # 5. Plot
        plt.figure(figsize=(figsize, figsize))
        plt.imshow(figure, cmap='gray')
        plt.axis('off')
        plt.title(f'VAE Latent Space Grid ({grid_size}x{grid_size})')
        plt.show()