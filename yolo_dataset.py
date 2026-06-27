"""
数据集类，用于访问数据集
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import yaml
import math



class YoloDataset(Dataset):
    def __init__(self,split,img_transform=None,label_transform=None):#split:"train","val","test"
        with open("config.yaml",'r') as f:
            config=yaml.safe_load(f)
        self.img_folder=os.path.join("datasets",config["dataset_name"],"images",split)
        self.label_folder=os.path.join("datasets",config["dataset_name"],"labels",split)
        self.img_names=os.listdir(self.img_folder)

        if img_transform is None:#默认适应模型输入
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((416, 416))
            ])
        elif img_transform=="no_transform":#不变换,返回原始图片
            self.img_transform=None
        else:#自定义变换
            self.img_transform=img_transform

        if label_transform is None:#默认适应模型输出
            self.label_transform=create_yolo_target
        elif label_transform=="no_transform":#不变换,返回dict
            self.label_transform=None
        else:#自定义变换
            self.label_transform=label_transform

        #初始化yaml内的nc与cls_list字段
        cls_path=os.path.join("datasets",config["dataset_name"],"classes.txt")
        cls_list=[]
        nc=0
        with open(cls_path) as f:
            cls_content = f.read()
            cls_names = cls_content.strip().split("\n")
            for cls_name in cls_names:
                cls_list.append(cls_name)
                nc+=1
        config["nc"]=nc
        config["cls_list"]=cls_list
        with open("config.yaml",'w') as f:
            yaml.dump(config,f,default_flow_style=None,sort_keys=False)#写回yaml

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self,idx):
        img_name=self.img_names[idx]
        img_path=os.path.join(self.img_folder,img_name)
        img=Image.open(img_path)
        #解析label
        label_name=img_name.split(".")[0]+".txt"
        label_path=os.path.join(self.label_folder,label_name)
        bboxes=[]
        clses=[]
        with open(label_path) as f:
            label_content=f.read()
            obj_infos=label_content.strip().split("\n")
            for obj_info in obj_infos:
                info_list=obj_info.strip().split(" ")
                cls=int(info_list[0])
                x_c=float(info_list[1])
                y_c=float(info_list[2])
                w=float(info_list[3])
                h=float(info_list[4])
                clses.append(cls)
                bboxes.append([x_c,y_c,w,h])
        obj_num=len(bboxes)#目标个数
        clses=torch.tensor(clses)#类别
        bboxes=torch.tensor(bboxes)#真实框
        label={
            "obj_num":obj_num,
            "clses":clses,
            "bboxes":bboxes,
        }
        if self.img_transform is not None:
            img=self.img_transform(img)
        if self.label_transform is not None:
            label=self.label_transform(label)
        return img,label
"""
默认返回的label结构(dict):
label={
        "obj_num":#目标个数
        "clses":#类别
        "bboxes":#预测框
    }
"""
def create_yolo_target(label):#制作target,形状:[13,13,5,(4+1+C)],其中(4+1+C)包括(tx,ty,tw,th,conf,C个onehot)
    #预测框bx=cx+σ(tx),by=cy+σ(ty),bw=pw*e^tw,bh=ph*e^th,其中(cx,cy),(pw,ph)分别为网格左上角xy坐标与锚框宽高
    clses=label["clses"]
    bboxes=label["bboxes"]
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    nc=config["nc"]
    anchors=config["anchors"]

    target=torch.zeros((13,13,5,5+nc))#初始化target形状,并且置0
    for cls,bbox in zip(clses,bboxes):
        x_c,y_c,bw,bh=bbox
        #匹配最佳形状的锚框,假设中心与真实框一致
        best_iou=0#最佳iou
        best_anchor_idx=-1#最佳锚框序号
        pw=-1;ph=-1#最佳锚框的宽、长
        for idx,(anchor_w,anchor_h) in enumerate(anchors):
            S_inter=min(anchor_w,bw)*min(anchor_h,bh)
            S_union=anchor_w*anchor_h+bw*bh-S_inter
            iou=S_inter/S_union#形状iou,无关cx,cy
            if iou>best_iou:
                best_iou=iou
                best_anchor_idx=idx
                pw=anchor_w
                ph=anchor_h
        #真实框在13*13坐标系的中心x,y坐标(未归一)
        bx=x_c*13
        by=y_c*13
        #负责预测网格的x,y坐标(13*13坐标系中)
        cx=int(bx)
        cy=int(by)
        #倒推真实框的σ(tx),σ(ty),即yolov1网格内的x_offset,y_offset,用此项计算损失
        x_offset=bx-cx
        y_offset=by-cy
        #倒推真实框的tw,th,用此项计算损失
        tw=math.log(bw/pw)
        th=math.log(bh/ph)
        #填充对应位置
        target[cy,cx,best_anchor_idx,0]=x_offset
        target[cy,cx,best_anchor_idx,1]=y_offset
        target[cy,cx,best_anchor_idx,2]=tw
        target[cy,cx,best_anchor_idx,3]=th
        target[cy,cx,best_anchor_idx,4]=1.0#置信度
        target[cy,cx,best_anchor_idx,5+cls]=1.0#类别onehot编码

    return target



if __name__=="__main__":
    train_dataset=YoloDataset(split="train",label_transform="no_transform")
    val_dataset=YoloDataset(split="val",label_transform="no_transform")
    print(len(train_dataset))
    print(len(val_dataset))
    # img,label=train_dataset[0]
    # print(img.shape)
    # print(label)