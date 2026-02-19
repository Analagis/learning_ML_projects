import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class SupervisedVAELoss(nn.Module):
    def forward(self, x, data, h_mean, h_log_var):
        bce = F.binary_cross_entropy(x, data, reduction='sum')
        kld = -0.5 * torch.sum(1 + h_log_var - h_mean.pow(2) - h_log_var.exp())
        return bce + kld
    
class SupervisedVAE(nn.Module):
    """
    Supervised Variational AutoEncoder.
    Добавляет классификацию цифр (0-9) поверх латентного пространства z.
    """
    def __init__(self, input_dim, latent_dim=1, n_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        encoder_input_dim = input_dim + n_classes

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        self.mean_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        decoder_input_dim = latent_dim + n_classes

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

        self.device = 'cuda'
    
    def encode(self, x, y):
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float()
        encoder_input = torch.cat([x, y_onehot], dim=1).to(self.device)
        h = self.encoder(encoder_input)
        return self.mean_layer(h), self.logvar_layer(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float()
        decoder_input = torch.cat([z, y_onehot], dim=1).to(self.device)
        return self.decoder(decoder_input)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar, z
    
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
        loss_func = SupervisedVAELoss()
        
        for _e in range(epochs):
            loss_mean = 0
            lm_count = 0
        
            train_tqdm = tqdm(train_loader, leave=True)
            for x_train, y_train in train_tqdm:
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                recon_batch, mu, logvar, z = self(x_train, y_train)
                loss = loss_func(recon_batch, x_train, mu, logvar, )
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                lm_count += 1
                loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
                train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

    def plot_latent_grid(self, latent_points=15, n_classes=10, limit=3):
        self.eval()

        z_values = np.linspace(-limit, limit, latent_points)
        figure = np.zeros((28 * n_classes, 28 * latent_points))
        
        with torch.no_grad():
            for i, digit in enumerate(range(n_classes)):
                for j, z_val in enumerate(z_values):
                    z = torch.FloatTensor([[z_val]]).to(self.device)
                    y = torch.LongTensor([digit]).to(self.device)

                    recon = self.decode(z, y).cpu().numpy().reshape(28, 28)

                    figure[i*28:(i+1)*28, j*28:(j+1)*28] = recon

        plt.figure(figsize=(20, 12))
        plt.imshow(figure, cmap='gray')

        plt.xticks(np.arange(14, 28*latent_points, 28), 
                [f'{z:.1f}' for z in z_values], fontsize=10)
        plt.yticks(np.arange(14, 28*n_classes, 28), 
                [f'Digit {i}' for i in range(n_classes)], fontsize=10)
        
        plt.xlabel('Скрытое измерение', fontsize=14)
        plt.ylabel('Цифровой класс', fontsize=14)
        plt.title('Генерация CVAE: 15 значений 10 разрядных классов', fontsize=16)
        plt.colorbar(label='Pixel Intensity')
        plt.tight_layout()
        plt.show()