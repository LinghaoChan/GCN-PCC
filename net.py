import torch
import torch.nn as nn
import torch.nn.functional as F
from getkNN import *


class GCNPCNET(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GCNPCNET, self).__init__()
        self.args = args
        self.k = args['k']
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args['emb_dims'])

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args['emb_dims'], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args['emb_dims']*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args['dropout'])
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args['dropout'])
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        
        x_tmp = get_graph_feature(x, k=self.k)
        x_tmp = torch.sum(x_tmp, dim=3).unsqueeze(dim=3)

        x = self.conv1(x_tmp)
        # print(x.size())
        x1 = x[:,:,:,0]
        # print(x1.size())

        x_tmp = get_graph_feature(x1, k=self.k)
        x_tmp = torch.sum(x_tmp, dim=3).unsqueeze(dim=3)
        x = self.conv2(x_tmp)
        x2 = x[:,:,:,0]

        x_tmp = get_graph_feature(x2, k=self.k)
        x_tmp = torch.sum(x_tmp, dim=3).unsqueeze(dim=3)
        x = self.conv3(x_tmp)
        x3 = x[:,:,:,0]

        x_tmp = get_graph_feature(x3, k=self.k)
        x_tmp = torch.sum(x_tmp, dim=3).unsqueeze(dim=3)
        x = self.conv4(x_tmp)
        x4 = x[:,:,:,0]

        x = torch.cat((x1, x2, x3, x4), dim=1)


        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x