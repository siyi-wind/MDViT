import torch.nn as nn
import numpy as np
import torch
import sys

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')


def Decomposition(model_name='AE', features=None, hidden_dim=64):
    features = torch.from_numpy(features).cuda()
    in_dim = features.shape[0]
    if model_name == 'AE':
        model = AutoEncoder(in_dim,hidden_dim)
    optimizer = torch.nn.


class AutoEncoder(nn.Module):
    '''
    AutoEncoder do decomposition
    '''
    def __init__(self, in_dim, h_dim):
        super(AutoEncoder,self).__init__()
        
        self.encoder = nn.Sequential([
            nn.Linear(in_dim,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,h_dim),
            nn.ReLU(inplace=True),
        ])

        self.decoder = nn.Sequential([
            nn.Linear(h_dim,128),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, in_dim),
            nn.Sigmoid(),
        ])
    
    def forward(x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



    



if __name__ == '__main__':
    x = np.random.rand(4000,512)
    model, y = AE(x)

