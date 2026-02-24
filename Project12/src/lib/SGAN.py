import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

img_shape = (1, 28, 28)

class Conditionalmodel_gen(nn.Module):
    """
    Генератор для Conditional GAN
    """
    def __init__(self, latent_dim = 100):
        super(Conditionalmodel_gen, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_emb = self.label_emb(labels)
        gen_input = torch.cat((z, label_emb), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class Conditionalmodel_dis(nn.Module):
    """
    Дискриминатор для Conditional GAN
    """
    def __init__(self):
        super(Conditionalmodel_dis, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + 10, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels)
        d_in = torch.cat((img_flat, label_emb), -1)
        validity = self.model(d_in)
        return validity
    
def train_SGAN(train_data, epochs, hidden_dim, lr = 0.001):
    device = torch.device("cuda")
    model_gen = Conditionalmodel_gen().to(device)
    model_dis = Conditionalmodel_dis().to(device)

    optimizer_gen = torch.optim.Adam(params=model_gen.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(params=model_dis.parameters(), lr=lr, betas=(0.5, 0.999))

    loss_func = nn.BCELoss()

    loss_gen_lst = []
    loss_dis_lst = []

    model_gen.train()
    model_dis.train()

    for _e in range(epochs):
        loss_mean_gen = 0
        loss_mean_dis = 0
        lm_count = 0
    
        train_tqdm = tqdm(train_data, leave=True)
        for x_train, y_train in train_tqdm:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            batch_size = x_train.size(0)
    
            h = torch.normal(mean=torch.zeros((batch_size, hidden_dim)), std=torch.ones((batch_size, hidden_dim)))
            h = h.to(device)

            optimizer_dis.zero_grad()
            real_preds = model_dis(x_train, y_train)
            d_real_loss = loss_func(real_preds, torch.ones(batch_size, 1).to(device))
            
            labels_fake = torch.randint(0, 10, (batch_size,)).to(device)
            fake_imgs = model_gen(h, labels_fake)
            fake_preds = model_dis(fake_imgs.detach(), labels_fake)
            d_fake_loss = loss_func(fake_preds, torch.zeros(batch_size, 1).to(device))
            
            loss_dis = d_real_loss + d_fake_loss
            loss_dis.backward()
            optimizer_dis.step()

            optimizer_gen.zero_grad()
            fake_preds_g = model_dis(fake_imgs, labels_fake)
            loss_gen = loss_func(fake_preds_g, torch.ones(batch_size, 1).to(device))
            loss_gen.backward()
            optimizer_gen.step()
    
            lm_count += 1
            loss_mean_gen = 1/lm_count * loss_gen.item() + (1 - 1/lm_count) * loss_mean_gen
            loss_mean_dis = 1/lm_count * loss_dis.item() + (1 - 1/lm_count) * loss_mean_dis
    
            train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean_gen={loss_mean_gen:.3f}, loss_mean_dis={loss_mean_dis:.3f}")
    
        loss_gen_lst.append(loss_mean_gen)
        loss_dis_lst.append(loss_mean_dis)

    return model_gen, loss_gen_lst, loss_dis_lst

def SGAN_plot_latent_grid(generator, hidden_dim, grid_size=15, n_classes=10, limit=3):
    device = torch.device("cuda")
    generator.eval()

    torch.manual_seed(42)
    z_values = np.linspace(-limit, limit, grid_size)
    figure = np.zeros((28 * n_classes, 28 * grid_size))
    z_grid = torch.randn(15, hidden_dim).to(device)
    
    with torch.no_grad():
        for i, digit in enumerate(range(n_classes)):
            for j, _ in enumerate(z_values):
                label = torch.full((15,), digit, dtype=torch.long).to(device)

                fake_imgs = generator(z_grid, label)
                fake_imgs = 0.5 * fake_imgs + 0.5

                figure[i*28:(i+1)*28, j*28:(j+1)*28] = fake_imgs[j].squeeze().cpu().numpy()

    plt.figure(figsize=(20, 12))
    plt.imshow(figure, cmap='gray')

    plt.xticks(np.arange(14, 28*grid_size, 28), 
            [f'{z:.1f}' for z in z_values], fontsize=10)
    plt.yticks(np.arange(14, 28*n_classes, 28), 
            [f'Digit {i}' for i in range(n_classes)], fontsize=10)
    
    plt.xlabel('Скрытое измерение', fontsize=14)
    plt.ylabel('Цифровой класс', fontsize=14)
    plt.title('Генерация CGAN: 15 значений 10 разрядных классов', fontsize=16)
    plt.colorbar(label='Pixel Intensity')
    plt.tight_layout()
    plt.show()