# model
import torch
import torch.nn as nn
'''
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=4096):
        return input.view(input.size(0), size, 1, 1)

class AE(nn.Module):
    def __init__(self, h_dim=4096, z_dim=20):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Flatten()
        )

        self.fc1 = nn.Linear(32 * 32 * 4, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, 32 * 32)
 
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x) # [64, 4096]
        x1 = self.fc1(x1)
        # print(x1.shape)
        # x1 = x1.view(x1.size(0), -1)
        # Flatten()
        print(x1.shape)
        z, mu , logvar = self.bottleneck(x1)
        z = self.fc3(z)
        z = self.fc4(z)
        print(z.shape)

        x  = self.decoder(z)
        return x1, x, mu, logvar

    def bottleneck(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
'''

# best 1
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)  # [64, 3, 32, 32]
        x  = self.decoder(x1) # [64, 256, 4, 4]
        return x1, x
'''
# 05201140
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            # nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), # added layer
            # nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            # nn.ConvTranspose2d(64, 16, 13, stride=1), # added layer
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 3, 17, stride=1),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)  # [64, 3, 32, 32]
        x  = self.decoder(x1) # [64, 256, 4, 4]
        return x1, x
'''