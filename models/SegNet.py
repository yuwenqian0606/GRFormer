1# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch.nn.functional as F
from  models.build import MODELS

   
@MODELS.register_module()
class SegNet(torch.nn.Module):
    def __init__(self, num_classes=2,num_point=16384):
        super(SegNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=(1, 1), padding=(0, 0))
        self.conv5 = torch.nn.Conv2d(128, 1024, kernel_size=(1, 1), padding=(0, 0))
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(num_point, 1), padding=(0, 0))
        self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.conv6 = torch.nn.Conv2d(1152, 512, kernel_size=(1, 1), padding=(0, 0))
        self.conv7 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), padding=(0, 0))
        self.conv8 = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), padding=(0, 0))
        self.dropout = torch.nn.Dropout(p=0.3)
        self.num_point = num_point

    def get_seg_loss(self, pred, label):
        """
        pred: Tensor of shape (B, N, 2) - model predictions (logits)
        label: Tensor of shape (B, N) - ground truth labels (class indices)
        """
        # Reshape pred and label to match CrossEntropyLoss requirements
        pred = pred.view(-1, pred.size(-1))  # (B*N, 2)
        label = label.view(-1)  # (B*N)
         # 确保 label 是 LongTensor 类型
        if label.dtype != torch.long:
            label = label.long()
        print("label:",label)
        # Define the CrossEntropyLoss criterion
        criterion = torch.nn.CrossEntropyLoss()
        
        # Compute the loss
        loss = criterion(pred, label)
        return loss

    def forward(self, points):
        batch_size = points.size(0)
        num_point = points.size(1)

        points = points.unsqueeze(-1)  # Add channel dimension
        print("points:",points.size())  # torch.Size([batch_size, 16384, 3, 1])
        points = points.permute(0, 3, 1,2)  # B,1,16384,3
        points = F.relu(self.conv1(points)) # B,64,N,1
        points = F.relu(self.conv2(points)) # B,64,N,1
        points = F.relu(self.conv3(points)) # B,64,N,1
        points = F.relu(self.conv4(points)) # B,128,N,1
        points_feat1 = F.relu(self.conv5(points))
        # print("points_feat1:",points_feat1.size())  # torch.Size([batch_size,1024, 16384, 1])

        pc_feat1 = self.maxpool(points_feat1)
        # print("pc_feat1:",pc_feat1.size())  # torch.Size([batch_size, 1024, 1, 1])
        pc_feat1 = pc_feat1.view(batch_size, -1)
        # print("pc_feat11:",pc_feat1.size())  # torch.Size([batch_size, 1024])

        pc_feat1 = F.relu(self.fc1(pc_feat1))
        pc_feat1 = F.relu(self.fc2(pc_feat1))
        # print("pc_feat111:",pc_feat1.size())  # torch.Size([batch_size, 128])
        
        pc_feat1_expand = pc_feat1.view(batch_size, 1, 1, -1).repeat(1, num_point, 1, 1)
        # print("pc_feat1_expand:",pc_feat1_expand.size())  # torch.Size([batch_size, 16384, 1, 128])
        points_feat1 = points_feat1.permute(0, 2,3,1)  # B,16384,1,1024
        # print("pointssssssssss:",points_feat1.size())  # torch.Size([batch_size, 16384, 1, 1024])
        points_feat1_concat = torch.cat([points_feat1, pc_feat1_expand], dim=3)
        # print("points_feat1_concat:",points_feat1_concat.size())  # torch.Size([batch_size, 16384,1, 1152])
        
        points_feat1_concat=points_feat1_concat.permute(0, 3, 1, 2)  # B,1152,16384,1
        net = F.relu(self.conv6(points_feat1_concat))
        # print("net:",net.size())  # torch.Size([batch_size, 512, 16384, 1])
        net = F.relu(self.conv7(net))
        net = self.dropout(net)
        net = self.conv8(net)
        # print("net222:",net.size())  # torch.Size([batch_size, 2, 16384, 1])
        net=net.permute(0, 2, 3, 1)  # B,16384,1,2
        net = net.squeeze(2)
        # print("net333:",net.size()) # torch.Size([batch_size,16384,2])

        return net