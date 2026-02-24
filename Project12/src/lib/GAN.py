import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
    
class Generator(nn.Module):
    """
    Генератор
    """
    def __init__(self, latent_dim=2):
        super().__init__()

        self.model_gen = nn.Sequential(
            nn.Linear(latent_dim, 512*7*7, bias=False),
            nn.ELU(),
            nn.BatchNorm1d(512*7*7),
            nn.Unflatten(1, (512, 7, 7)),
            nn.Conv2d(512, 256, 5, 1, padding='same', bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 5, 1, padding='same', bias=False),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model_gen(x)
    
class Discriminator(nn.Module):
    """
    Дискриминатор
    """
    def __init__(self):
        super().__init__()

        self.model_dis = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, padding=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, 2, padding=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
        )

    def forward(self, x):
        return self.model_dis(x)

def train_GAN(train_data, epochs, hidden_dim, batch_size, lr = 0.001):
    device = torch.device("cuda")
    model_gen = Generator().to(device)
    model_dis = Discriminator().to(device)

    optimizer_gen = torch.optim.Adam(params=model_gen.parameters(), lr=lr)
    optimizer_dis = torch.optim.Adam(params=model_dis.parameters(), lr=lr)

    loss_func = nn.BCEWithLogitsLoss()
    targets_0 = torch.zeros(batch_size, 1).to(device)
    targets_1 = torch.ones(batch_size, 1).to(device)

    loss_gen_lst = []
    loss_dis_lst = []

    model_gen.train()
    model_dis.train()

    for _e in range(epochs):
        loss_mean_gen = 0
        loss_mean_dis = 0
        lm_count = 0
    
        train_tqdm = tqdm(train_data, leave=True)
        for x_train, _ in train_tqdm:
            x_train = x_train.to(device)
    
            h = torch.normal(mean=torch.zeros((batch_size, hidden_dim)), std=torch.ones((batch_size, hidden_dim)))
            h = h.to(device)
    
            img_gen = model_gen(h)
            fake_out = model_dis(img_gen)
    
            loss_gen = loss_func(fake_out, targets_1)
    
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
    
            img_gen = model_gen(h)
            fake_out = model_dis(img_gen)
            real_out = model_dis(x_train)
    
            outputs = torch.cat([real_out, fake_out], dim=0).to(device)
            targets = torch.cat([targets_1, targets_0], dim=0).to(device)
    
            loss_dis = loss_func(outputs, targets)
    
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
    
            lm_count += 1
            loss_mean_gen = 1/lm_count * loss_gen.item() + (1 - 1/lm_count) * loss_mean_gen
            loss_mean_dis = 1/lm_count * loss_dis.item() + (1 - 1/lm_count) * loss_mean_dis
    
            train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean_gen={loss_mean_gen:.3f}, loss_mean_dis={loss_mean_dis:.3f}")
    
        loss_gen_lst.append(loss_mean_gen)
        loss_dis_lst.append(loss_mean_dis)

    return model_gen, loss_gen_lst, loss_dis_lst

class GradientReversalFunction(torch.autograd.Function):
    """
    Autograd функция для обратного градиента
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(x)
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    """
    Модуль-обертка для GradientReversalFunction
    """
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

class GANWithGradientReversal(nn.Module):
    """
    Полная модель GAN со слоем обратного градиента
    """
    def __init__(self, latent_dim=2, grad_reverse_lambda=0.5):
        super(GANWithGradientReversal, self).__init__()
        self.latent_dim = latent_dim
        self.grad_reverse_lambda = grad_reverse_lambda

        self.generator = Generator(latent_dim)

        self.grl = GradientReversalLayer(lambda_=grad_reverse_lambda)

        self.discriminator = Discriminator()
    
    def forward(self, z, real_data=None):
        fake_data = self.generator(z)
        real_disc_out = self.discriminator(real_data)
        
        fake_data_with_grl = self.grl(fake_data)
        fake_disc_out = self.discriminator(fake_data_with_grl)
        
        return real_disc_out, fake_disc_out, fake_data

def gan_loss_with_grl(real_output, fake_output):
    real_loss = F.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output))
    
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output))
    
    return real_loss + fake_loss


def train_GAN_gr(train_data, epochs, hidden_dim, lr = 0.001):
    device = torch.device("cuda")
    model = GANWithGradientReversal(
        latent_dim=hidden_dim, 
        grad_reverse_lambda=15
    ).to(device)    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []
    real_scores = []
    fake_scores = []

    for epoch in range(epochs):
        train_tqdm = tqdm(train_data, leave=True)
        for batch_idx, (real_imgs, _) in enumerate(train_tqdm):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            z = torch.randn(batch_size, 2).to(device)
            real_output, fake_output, _ = model(z, real_imgs)
            
            loss = gan_loss_with_grl(real_output, fake_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                d_losses.append(loss.item())
                real_scores.append(real_output.mean().item())
                fake_scores.append(fake_output.mean().item())

                with torch.no_grad():
                    g_loss = F.binary_cross_entropy_with_logits(
                        fake_output, torch.ones_like(fake_output))
                    g_losses.append(g_loss.item())

            train_tqdm.set_description(f"Epoch [{epoch+1}/{epochs}], loss_mean_gen={loss:.3f}")

        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Total Loss: {d_losses[-1]:.4f} | '
                f'G Loss: {g_losses[-1]:.4f} | Real: {real_scores[-1]:.3f} | '
                f'Fake: {fake_scores[-1]:.3f}')

    return model, g_losses, d_losses


def GAN_plot_latent_grid(model_gen):
    model_gen.eval()
    n = 7
    device = 'cuda'
    grid_size = 15

    figure = np.zeros((28 * grid_size, 28 * grid_size))

    for r, i in enumerate(range(-n, n+1)):
        for c, j in enumerate(range(-n, n+1)):
            h = torch.tensor([[1 * i / n, 1 * j / n]], dtype=torch.float32)
            predict = model_gen(h.to(device))
            predict = predict.detach().squeeze()
            dec_img = predict.cpu().numpy()

            figure[r*28:(r+1)*28, c*28:(c+1)*28] = dec_img

    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title(f'Сгенерированные GAN цифры: {15}x{15}', fontsize=16)
    plt.tight_layout()
    plt.show()