import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class API_Net(nn.Module):
    def __init__(self):
        super(API_Net, self).__init__()

        # resnet50
        resnet50 = models.resnet50(pretrained=True)
        layers = list(resnet50.children())[:-2]

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.fc = nn.Linear(2048, 45)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.map3 = nn.Linear(2048 * 2, 2048)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

    def forward(self, images, targets=None, flag='train'):
        conv_out = self.conv(images)

        # #ECA
        b, c, h, w = conv_out.size()
        conv_out11 = self.avg_pool(conv_out)
        conv_out12 = self.conv1(conv_out11.squeeze(-1).transpose(-1, -2))
        conv_out13 = conv_out12.transpose(-1, -2).unsqueeze(-1)
        conv_out14 = self.sigmoid(conv_out13)
        conv_out15 = conv_out * conv_out14.expand_as(conv_out)
        pool_out2 = self.avg(conv_out15).squeeze()

        pool_out = self.avg(conv_out).squeeze()

        if flag == 'train':
            intra_pairs, inter_pairs, \
                    intra_labels, inter_labels = self.get_pairs(pool_out,targets
                                                               )


            # # # # ECA1
            features3 = torch.cat([conv_out[intra_pairs[:, 0]], conv_out[inter_pairs[:, 0]]], dim=0)
            features4 = torch.cat([conv_out[intra_pairs[:, 1]], conv_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features3, features4], dim=1)


            b, c, h, w = mutual_features.size()
            conv_out1 = self.avg_pool(mutual_features)
            conv_out2 = self.conv1(conv_out1.squeeze(-1).transpose(-1, -2))
            conv_out3 = conv_out2.transpose(-1, -2).unsqueeze(-1)
            conv_out4 = self.sigmoid(conv_out3)
            conv_out5 = mutual_features * conv_out4.expand_as(mutual_features)
            pool_out1 = self.avg(conv_out5).squeeze()
            a_mut = self.map3(pool_out1)


            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            features5 = torch.cat([pool_out2[intra_pairs[:, 0]], pool_out2[inter_pairs[:, 0]]], dim=0)
            features6 = torch.cat([pool_out2[intra_pairs[:, 1]], pool_out2[inter_pairs[:, 1]]], dim=0)


            map3_out = self.sigmoid(a_mut)
            features1_mut = torch.mul(a_mut, features1)
            features2_mut = torch.mul(a_mut, features2)


            logit1_mut = self.fc(self.drop(features1_mut))
            logit2_mut = self.fc(self.drop(features2_mut))
            logit1_self = self.fc(self.drop(features5))
            logit2_self = self.fc(self.drop(features6))


            return logit1_self, logit2_self, labels1, labels2, logit1_mut, logit2_mut



        elif flag == 'val':
            return self.fc(pool_out)



    def get_pairs(self, embeddings, labels):
        distance_matrix = pdist(embeddings).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]

        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        # #intra_S
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        # #inter_S
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])



        for i in range(embeddings.shape[0]):

            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]


        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels


















