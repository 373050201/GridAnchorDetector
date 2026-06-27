"""
损失类，用于定义损失函数
"""
import torch
import yaml
from torch import nn
import random



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        with open("config.yaml",'r') as f:
            config=yaml.safe_load(f)
        self.nc=config["nc"]#类别数
        self.location_loss=nn.MSELoss()#位置损失,均方差
        self.confidence_loss=nn.BCELoss(reduction="none")#置信度损失,二元交叉熵
        self.class_loss=nn.CrossEntropyLoss()#类别损失,交叉熵
        self.lambda_obj=config["lambda_obj"]#置信度损失中正样本权重
        self.lambda_noobj=config["lambda_noobj"]#置信度损失中负样本权重
        self.lambda_lct_loss=config["lambda_lct_loss"]#总损失中位置损失权重
        self.lambda_conf_loss=config["lambda_conf_loss"]#总损失中置信度损失权重
        self.lambda_cls_loss=config["lambda_cls_loss"]#总损失中类别损失权重
    def forward(self,predict,target):
        #predict/target形状:[B,13,13,5,5+nc]
        obj_mask=target[...,4]>0.5#正样本bool掩码,shape:[B,13,13,5],满足条件则置为True
        if obj_mask.sum()==0:#若无正样本.返回损失0
            return torch.tensor(0.0)
        noobj_mask=~obj_mask#负样本bool掩码
        #bool索引,可理解为保留最后一个维度(此处最后一维大小为4),筛选前面所有维度的正确个数N,拼成shape为(N,4)的向量
        #位置损失,仅正样本计算
        lct_loss=self.location_loss(predict[...,0:4][obj_mask],target[...,0:4][obj_mask])
        #置信度损失,正负样本均计算
        conf_all=self.confidence_loss(predict[...,4],target[...,4])
        conf_pos=conf_all[obj_mask].sum()#正样本置信度损失总和
        conf_neg=conf_all[noobj_mask].sum()#负样本置信度损失总和
        N=obj_mask.numel()#样本总数
        conf_loss=(self.lambda_obj*conf_pos+self.lambda_noobj*conf_neg)/N
        #类别损失,仅正样本计算
        cls_loss=self.class_loss(predict[...,5:5+self.nc][obj_mask],target[...,5:5+self.nc][obj_mask])
        #总损失
        sum_loss=self.lambda_lct_loss*lct_loss+self.lambda_conf_loss*conf_loss+self.lambda_cls_loss*cls_loss
        return sum_loss



class OldLoss(nn.Module):
    def __init__(self):
        super(OldLoss, self).__init__()
        with open("config.yaml",'r') as f:
            config=yaml.safe_load(f)
        self.nc=config["nc"]#类别数
        self.location_loss=nn.MSELoss()#位置损失,均方差
        self.confidence_loss=nn.BCELoss()#置信度损失,二元交叉熵
        self.class_loss=nn.CrossEntropyLoss()#类别损失,交叉熵
    def forward(self,predict,target):
        #predict/target形状:[B,13,13,5,5+nc]
        obj_mask=target[...,4]>0.5#正样本掩码
        if obj_mask.sum()==0:#若无正样本.返回损失0
            return torch.tensor(0.0)
        lct_loss=self.location_loss(predict[...,0:4][obj_mask],target[...,0:4][obj_mask])#正样本才计算位置损失
        conf_loss=self.confidence_loss(predict[...,4],target[...,4])#所有样本计算置信度损失
        cls_loss=self.class_loss(predict[...,5:5+self.nc][obj_mask],target[...,5:5+self.nc][obj_mask])#正样本才计算类别损失
        sum_loss=5.0*lct_loss+1.0*conf_loss+1.0*cls_loss#权重为5:1:1
        return sum_loss



if __name__=="__main__":
    print()
    # B=2
    # pred=torch.randn(B,13,13,5,8)
    # tgt=torch.randn(B,13,13,5,8)
    # pred[...,4]=torch.tensor(random.random())
    # tgt[...,4]=torch.tensor(random.random())
    # loss=Loss()
    # lossVal=loss(pred,tgt)
    # print(lossVal)