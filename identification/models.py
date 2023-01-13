import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet152': [resnet152, 2048]
}


class SupConResNet(nn.Module):
    """encoder + projection head"""
    def __init__(self, encoder='resnet50', head='mlp', feat_dim=128, load_pt_encoder=False):
        super(SupConResNet, self).__init__()
        model_fun, dim_en = model_dict[encoder]
        self.encoder = model_fun(pretrained=True) if load_pt_encoder else model_fun()
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        self.dim_en = dim_en
        self.dim_ft = feat_dim
        if head == 'linear':
            self.head = nn.Linear(dim_en, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_en, dim_en),
                nn.ReLU(inplace=True),
                nn.Linear(dim_en, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def encode(self, x):
        """return the encoded features by ResNet of size (B, D_en)"""
        x = self.encoder(x)
        while x.dim() > 2:
            x = torch.squeeze(x, dim=2)
        return x
    
    def project(self, x):
        """return the projected features by ResNet and projection head of size (B, D_ft)"""
        x = self.encode(x)
        x = self.head(x)
        return x
    
    def forward(self, x):
        """return the normalized and projected features of size (B, D_ft)"""
        x = self.project(x)
        x = F.normalize(x, dim=1)
        return x



class LMCLResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, encoder='resnet50', load_pt_encoder=False, num_classes=17):
        super(LMCLResNet, self).__init__()
        model_fun, dim_en = model_dict[encoder]
        self.encoder = model_fun(pretrained=True) if load_pt_encoder else model_fun()
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        self.weight = nn.Linear(dim_en, num_classes)
        nn.init.xavier_uniform_(self.weight)
        
    
    def encode(self, x):
        """return the encoded features by ResNet of size (B, D_en)"""
        x = self.encoder(x)
        while x.dim() > 2:
            x = torch.squeeze(x, dim=2)
        return x
    
    def forward(self, x):
        """return the cosine similarity logits of size (B, C)"""
        x = self.encode(x)  # (B, D_en)
        logits = torch.mm(x, self.weight)   # (B, C)
        x_norm = torch.norm(x, p=2, dim=1)  # normalize over each input feature
        w_norm = torch.norm(self.weight, p=2, dim=0)    # normalize over each class feature
        logits = logits / torch.outer(x_norm, w_norm)
        return logits



class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=17):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
    
class MLPClassifier(nn.Module):
    """MLP classifier"""
    def __init__(self, name='resnet50', num_classes=17, projection_size=4096):
        super(MLPClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.mlp = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, projection_size)
    )
        self.fc = nn.Linear(projection_size, num_classes)
    
    def forward(self, features):
        return self.fc(self.mlp(features))

class SimSiamClassifier(nn.Module):
    """SimSiam classifier"""
    def __init__(self, name='resnet50', num_classes=17, projection_size=4096):
        super(SimSiamClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.mlp = nn.Sequential(
        nn.Linear(feat_dim, feat_dim, bias=False),
        nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, feat_dim, bias=False),
        nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )
        self.fc = nn.Linear(projection_size, num_classes)
    
    def forward(self, features):
        return self.fc(self.mlp(features))
        
