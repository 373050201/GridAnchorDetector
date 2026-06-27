"""
计算类，用于定义某些计算及函数
"""
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yolo_dataset import YoloDataset
import yaml
from torchvision.ops import box_iou
from PIL import Image,ImageDraw
from torchvision import transforms
import numpy as np
import time



def calculate_kmeans(dataset,write_yaml=False,k=5):#计算数据集的k种初始锚框,默认k=5
    sample=[]#存储每个bbox宽高[w,h]
    for _,label in dataset:
        for _,_,w,h in label["bboxes"]:
            sample.append([w.item(),h.item()])
    sample=torch.tensor(sample)
    kmeans=KMeans(n_clusters=k,random_state=42)#对宽高进行聚类,得到k个聚类中心,即k个锚框大小
    kmeans.fit(sample)
    anchors=kmeans.cluster_centers_
    anchors=anchors.tolist()
    anchors=[[round(w,4),round(h,4)] for w,h in anchors]#保留四位小数
    print(f"所有锚框：{anchors}")
    if write_yaml:#将anchors写入yaml
        with open("config.yaml","r") as f:
            config=yaml.safe_load(f)
        config["anchors"]=anchors
        with open("config.yaml","w") as f:
            yaml.dump(config,f,default_flow_style=None,sort_keys=False)
    return anchors



def calculate_iou(bbox1,bbox2):#计算两个bbox的iou,输入为[x_c,y_c,w,h]
    bbox1=torch.tensor(bbox1)
    bbox2=torch.tensor(bbox2)
    bbox1.unsqueeze_(0)
    bbox2.unsqueeze_(0)
    iou=box_iou(bbox1,bbox2,fmt="cxcywh").item()
    return iou



def calculate_avg_iou(dataset):#计算锚框与数据集目标框的平均形状iou(本地测试用)
    with open("config.yaml",'r') as f:
        config=yaml.safe_load(f)
    anchors=config["anchors"]
    iou_sum=0
    iou_cnt=0
    for _,label in dataset:
        bboxes=label["bboxes"]
        for _,_,t_w,t_h in bboxes:
            for a_w,a_h in anchors:
                inter=min(t_w,a_w)*min(t_h,a_h)
                union=t_w*t_h+a_w*a_h-inter
                iou_sum+=inter/union
                iou_cnt+=1
    avg_iou=iou_sum/iou_cnt
    return avg_iou



def visualize_bbox(img,bboxes,color="green"):#可视化bboxes,bboxes:[x_c,y_c,w,h,conf,cls]的列表
    if type(img)==torch.Tensor:
        img=transforms.ToPILImage()(img)#将img转化为PIL格式
    W,H=img.size
    with open("config.yaml",'r') as f:
        config=yaml.safe_load(f)
    cls_list=config["cls_list"]
    draw=ImageDraw.Draw(img)#创建Draw类的对象
    for bbox in bboxes:
        x_c,y_c,w,h,conf,cls=bbox
        x_max=W*(x_c+0.5*w)
        y_max=H*(y_c+0.5*h)
        x_min=W*(x_c-0.5*w)
        y_min=H*(y_c-0.5*h)
        draw.rectangle(xy=[x_min,y_min,x_max,y_max],outline=color,width=2)#绘制矩形
        draw.text(xy=(x_min,y_max),text=f"conf:{conf:.2f}",fill="red",font_size=12)#绘制置信度
        draw.text(xy=(x_min,y_min-25),text=f"{cls_list[cls]}",fill="green",font_size=20)#绘制类别
    plt.imshow(img)
    plt.axis("off")
    plt.show()



def nms(bboxes,iou_threshold=0.5):#非极大值抑制,去除预测框中的重复框,bboxes:[x_c,y_c,w,h,conf,cls]的列表
    bbox_cls={}#将bbox按类别分类,字典格式为{"cls":对应cls的bboxes的列表}
    for bbox in bboxes:
        cls=int(bbox[5])
        if cls not in bbox_cls:
            bbox_cls[cls]=[]
        bbox_cls[cls].append(bbox)
    all_remains=[]#所有最终被保留的框
    for cls in bbox_cls:#按类别nms
        bboxes=bbox_cls[cls]
        bboxes.sort(key=lambda x:x[4],reverse=True)#按置信度conf降序排序
        remains=[]#同组类别的保留框
        while len(bboxes)>0:
            best=bboxes.pop(0)#取出conf最高的框best
            remains.append(best)#保留best
            bboxes=[bbox for bbox in bboxes if calculate_iou(best[:4],bbox[:4])<iou_threshold]#暂时保留与best的iou低的框
        all_remains.extend(remains)
    return all_remains



def nms_vectorized(bboxes,iou_threshold=0.5):#非极大值抑制(向量化版),bboxes:[x_c,y_c,w,h,conf,cls]的列表
    if len(bboxes)==0:
        return []
    bboxes=torch.tensor(bboxes)#转换为tensor,shape:[N,6]
    all_cls=torch.unique(bboxes[:,5].long())#所有类别,shape:[nc]
    all_remains=[]#所有保留框
    for cls in all_cls:
        #选出类别为cls的所有bbox,存入bboxes_cls
        cls_mask=bboxes[:,5]==cls#假设有M个符合条件
        bboxes_cls=bboxes[cls_mask]#shape:[M,6]
        #按置信度降序排序
        col=bboxes_cls[:,4]#取出置信度一列
        order=col.argsort(descending=True)#得到降序对应的idx
        bboxes_cls=bboxes_cls[order]#降序排序
        #循环处理直到没有框剩余
        while len(bboxes_cls)>0:
            best=bboxes_cls[0:1]#取出conf最高的框best,shape:[1,6]
            all_remains.append(best[0].tolist())#保留best
            rest=bboxes_cls[1:]#除best外的剩余框,shape:[M-1,6]
            ious=box_iou(best[:,:4],rest[:,:4],fmt="cxcywh")#best与rest中每个bbox的iou,shape:[1,M-1]
            ious=ious[0]#shape:[M-1]
            keep_mask=ious<iou_threshold#shape:[M-1],假设其中有t个满足条件
            bboxes_cls=rest[keep_mask]#shape:[t,6]
    return all_remains



def calculate_mAP(all_preds,all_tgts,nc,iou_threshold=0.5):#计算mAP,all_preds/all_tgts:[img_id,x_c,y_c,w,h,conf,cls]的列表
    APs=[]#每个类别的AP
    for cls_id in range(nc):#按类别筛选
        preds=[pred for pred in all_preds if pred[-1]==cls_id]
        tgts=[tgt for tgt in all_tgts if tgt[-1]==cls_id]
        #预测框按置信度降序排序
        preds.sort(key=lambda x:x[5],reverse=True)
        #初始化用于记录的数据结构
        tp=np.zeros(len(preds))#数组,存储预测框是否为TP (True Positive:预测正确的预测框)
        fp=np.zeros(len(preds))#数组,存储预测框是否为FP (False Positive:预测错误的预测框)
        tgt_matched=[False]*len(tgts)#数组,存储真实框是否已被预测框匹配(要求一对一匹配)
        #为每个预测框匹配一个真实框,即找与此预测框iou最大的真实框
        for pred_idx,pred in enumerate(preds):
            best_iou=0.0#匹配的真实框与预测框的iou
            best_tgt_idx=-1#匹配的真实框索引
            for tgt_idx,tgt in enumerate(tgts):
                if pred[0]!=tgt[0] or tgt_matched[tgt_idx]==True:#若图不同或此真实框已被匹配,则跳过
                    continue
                iou=calculate_iou(pred[1:5],tgt[1:5])#计算iou
                if iou>best_iou:#更新为最佳真实框
                    best_iou=iou
                    best_tgt_idx=tgt_idx
            if best_iou>iou_threshold:#若iou大于阈值,判断预测准确
                tp[pred_idx]=1
                tgt_matched[best_tgt_idx]=True
            else:#否则判断预测错误
                fp[pred_idx]=1
        #计算精确率(Precision)和召回率(Recall)序列
        tp_cum=np.cumsum(tp)#累加的tp,fp序列
        fp_cum=np.cumsum(fp)#cumsum:累加函数,如:[1,2,3]->[1,3,6]
        precisions=tp_cum/(tp_cum+fp_cum+1e-10)#精确率序列(PR曲线纵坐标)
        recalls=tp_cum/(len(tgts)+1e-10)#召回率序列(PR曲线横坐标)
        #保证precisions单调不增
        for i in range(len(precisions)-2,-1,-1):
            precisions[i]=max(precisions[i],precisions[i+1])
        #计算此类别AP(PR曲线下面积)
        AP=0.0
        for i in range(len(precisions)):#模拟积分
            if i==0:
                AP+=precisions[i]*recalls[i]
                continue
            AP+=precisions[i]*(recalls[i]-recalls[i-1])
        APs.append(AP)
    #计算mAP
    mAP=np.mean(APs)#所有AP均值
    return mAP



if __name__=="__main__":
    print()
    test_bboxes = [
        # 类别0：两个高度重叠的框 (IoU > 0.5)
        [0.5, 0.5, 0.4, 0.4, 0.9, 0],  # 高置信度，应保留
        [0.48, 0.49, 0.42, 0.41, 0.81, 0],  # 与第一个框重叠，置信度较低，应被抑制

        # 类别0：另一个不重叠的框 (与前面的框IoU < 0.5)
        [0.1, 0.1, 0.2, 0.2, 0.71, 0],  # 远离前两个，应保留

        # 类别1：两个部分重叠的框 (IoU > 0.5)
        [0.3, 0.7, 0.3, 0.2, 0.88, 1],  # 高置信度，应保留
        [0.31, 0.695, 0.29, 0.19, 0.76, 1],  # 与上一个重叠，置信度较低，应被抑制

        # 类别1：一个孤立框
        [0.8, 0.8, 0.15, 0.15, 0.43, 1],  # 置信度低但无重叠，应保留（因为IoU阈值0.5，且无其他框）
    ]
    start1=time.time()
    print(nms(test_bboxes))
    end1=time.time()
    print(f"time:{end1-start1}")
    start2=time.time()
    print(nms_vectorized(test_bboxes))
    end2=time.time()
    print(f"time:{end2-start2}")