import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class VAELoss(nn.Module):
    def forward(self, x, y, h_mean, h_log_var):
        img_loss = torch.sum(torch.square(x - y), dim=-1)
        kl_loss = -0.5 * torch.sum(1 + h_log_var - torch.square(h_mean) - torch.exp(h_log_var), dim=-1)
        return torch.mean(img_loss + kl_loss)
    
class BaseVAE(nn.Module):
    """
    Базовый Variational AutoEncoder с ELBO loss.
    
    Args:
        input_dim (int): Размерность входных данных (например, 784 для MNIST).
        hidden_dims (list): Слои скрытых сетей encoder/decoder.
        z_dim (int): Размерность латентного пространства (по умолчанию 2).
        beta (float): Веса KL терма в ELBO (по умолчанию 1.0).
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64)
        )
 
        self.h_mean = nn.Linear(64, self.hidden_dim)
        self.h_log_var = nn.Linear(64, self.hidden_dim)
 
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

        self.device = 'cuda'
    
    def forward(self, x):
        enc = self.encoder(x)
 
        h_mean = self.h_mean(enc)
        h_log_var = self.h_log_var(enc)
 
        noise = torch.normal(mean=torch.zeros_like(h_mean), std=torch.ones_like(h_log_var)).to(self.device)
        h = noise * torch.exp(h_log_var / 2).to(self.device) + h_mean
        x = self.decoder(h)
 
        return x, h, h_mean, h_log_var

    def fit(self, 
            train_loader, 
            epochs: int = 50, 
            lr: float = 1e-3):
        """
        Обучение VAE с валидацией и логированием.
        
        Returns:
            list: Список средних loss по эпохам.
        """
        self.to(self.device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = VAELoss()
        
        for _e in range(epochs):
            loss_mean = 0
            lm_count = 0
        
            train_tqdm = tqdm(train_loader, leave=True)
            for x_train, y_train in train_tqdm:
                x_train = x_train.to(self.device)
                predict, _, h_mean, h_log_var = self(x_train)
                loss = loss_func(predict, x_train, h_mean, h_log_var)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                lm_count += 1
                loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
                train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")
    
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
            generated_images = self.decoder(z_tensor).cpu()  # [225, 784]
        
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