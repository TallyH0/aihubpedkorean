import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    def __init__(self, num_class, dim_feature, device=torch.device('cuda')):
        super().__init__()
        
        # self.center = nn.Parameter(torch.zeros(num_class, dim_feature), requires_grad=False).to(device)
        self.center = torch.zeros(num_class, dim_feature, requires_grad=False).to(device)
        self.num_class = num_class
        self.dim_feature = dim_feature
        self.alpha = 0.95
        self.margin = 1
        
    def forward(self, x, y):
        # x : [batch, dim_feature]
        # y : [batch]
        batch_size = y.shape[0]
        
        with torch.no_grad():
            y = torch.repeat_interleave(y, self.dim_feature, 0).reshape(-1, self.dim_feature)
            center_batch = torch.gather(self.center, 0, y)
            diff = -1 * (1 - self.alpha) * (center_batch - x)
            self.center.scatter_(0, y, diff)
            center_batch = torch.gather(self.center, 0, y)
        
        loss_center = (x - center_batch).pow(2).mean()
        feature_norm = x.pow(2).sum(dim=1, keepdim=True)
        center_norm = self.center.pow(2).sum(dim=1, keepdim=True)
        
        feature_norm_cast = torch.broadcast_to(feature_norm, (batch_size, batch_size))
        feature_norm_cast = feature_norm_cast + feature_norm_cast.transpose(1, 0)
        
        feature_dot_product = torch.matmul(x, x.transpose(1, 0))
        feature_diff = feature_norm_cast - 2 * feature_dot_product
        
        loss_push = F.relu(-feature_diff.mean() + loss_center + self.margin)
        
        feature_norm_cast2 = torch.broadcast_to(feature_norm, (batch_size, self.num_class))
        center_norm_cast = torch.broadcast_to(center_norm, (self.num_class, batch_size))
        
        feature_center_norm = feature_norm_cast2 + center_norm_cast.transpose(1, 0)
        feature_center_product = torch.matmul(x, self.center.transpose(1, 0))
        
        feature_center_diff = feature_center_norm - 2 * feature_center_product
        loss_gpush = F.relu(-feature_center_diff.mean() + 2 * loss_center + self.margin)
        
        return loss_center, loss_push, loss_gpush
