import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        # 先随机初始化特征中心，再训练更新
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))  # (20,1024)

    def forward(self, feature, labels):
        """
        Args:
            feature: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size). labels中记录的实际上是每个类别的索引
        """
        batch_size = feature.size(0)  # B
        """
        参考：https://blog.csdn.net/IT_forlearn/article/details/100022244
        xx(feature)经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，
        此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        yy会在最后进行转置的操作
        """
        # 计算矩阵feature和centers的欧式距离，只选取对应类别的distant值
        distmat = torch.pow(feature, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()  # (B,20)
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT,addmm_表示对原有数据进行修改
        distmat.addmm_(1, -2, feature, self.centers.t()) #(B,20)

        classes = torch.arange(self.num_classes).long()  # [0,1,2,....,19]
        if self.use_gpu: classes = classes.cuda()

        if labels.numel() > labels.size(0):
            mask = labels > 0
        else:
            #(B,)-->(B,1)-->(B,num_classes)
            labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)  # (B,20)
            mask = labels.eq(classes.expand(batch_size, self.num_classes).float()) # (B,20)

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]  
            value *= labels[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability clamp()函数可以限定dist内元素的最大最小范围
            dist.append(value)
        dist = torch.cat(dist)  #(B,)
        loss = dist.mean()

        return loss
    
