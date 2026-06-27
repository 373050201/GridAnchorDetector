"""
模型训练
评估标准:预测准确率acc
"""
from loss import Loss,OldLoss
from model import GridAnchorDetector
from yolo_dataset import YoloDataset
import yaml
from torch.utils.data import DataLoader
import torch
from calculate import calculate_iou,calculate_kmeans
import time



print("正在更新锚框...")
train_dataset=YoloDataset(split="train",label_transform="no_transform")
calculate_kmeans(train_dataset,write_yaml=True)

print("正在加载训练/验证集...")
batch_size=8
train_dataset=YoloDataset("train")
train_loader=DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
val_dataset=YoloDataset("val")
val_loader=DataLoader(val_dataset,batch_size,shuffle=True,drop_last=True)
with open("config.yaml",'r') as f:
    config=yaml.safe_load(f)
anchors=config["anchors"]
nc=config["nc"]
learning_rate=config["learning_rate"]

print("正在初始化模型...")
detector=GridAnchorDetector()
detector=detector.cuda()
loss=Loss()
loss=loss.cuda()
optimizer=torch.optim.Adam(detector.parameters(),lr=learning_rate)

start_time=time.time()#训练计时开始
total_step=0
epoch=100
best_accuracy=0#最佳准确率
best_epoch=0#最佳训练轮次
for i in range(epoch):
    print(f"第{i+1}轮训练开始...")
    detector.train()#训练模式
    total_train_loss=0
    for imgs,labels in train_loader:
        imgs=imgs.cuda()
        labels=labels.cuda()

        preds=detector(imgs)
        train_loss=loss(preds,labels)
        total_train_loss+=train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_step+=1
    print(f"此轮已累计训练了{total_step}次")
    print(f"训练集总损失：{total_train_loss:.6f}")

    detector.eval()#评估模式
    total_val_loss=0
    total_obj=0#目标数(准确来说乘了锚框数)
    total_correct=0#预测正确的锚框数
    count=0#早停计数
    patience=15#最多容忍patience个epoch没有改善
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs=imgs.cuda()
            labels=labels.cuda()

            preds=detector(imgs)
            val_loss=loss(preds,labels)
            total_val_loss+=val_loss
            #解码
            for b in range(batch_size):
                for grid_y in range(13):
                    for grid_x in range(13):
                        for anchor_idx in range(5):
                            if labels[b,grid_y,grid_x,anchor_idx,4]>0.5:
                                total_obj+=1
                                #类别解码
                                cls_pred=torch.argmax(preds[b,grid_y,grid_x,anchor_idx,5:5+nc]).item()
                                cls_tgt=torch.argmax(labels[b,grid_y,grid_x,anchor_idx,5:5+nc]).item()
                                if cls_pred!=cls_tgt:#若类别预测错误,则无需后续判断
                                    continue
                                #当前锚框宽高
                                anchor_w,anchor_h=anchors[anchor_idx]
                                #目标框bbox解码
                                tx_tgt,ty_tgt,tw_tgt,th_tgt=labels[b,grid_y,grid_x,anchor_idx,:4]
                                x_c_tgt=(grid_x+tx_tgt)/13.0#此处tx_tgt,ty_tgt已在tgt制作时归一化
                                y_c_tgt=(grid_y+ty_tgt)/13.0
                                w_tgt=anchor_w*torch.exp(tw_tgt)
                                h_tgt=anchor_h*torch.exp(th_tgt)
                                bbox_tgt=[x_c_tgt,y_c_tgt,w_tgt,h_tgt]
                                #预测框bbox解码
                                tx_pred,ty_pred,tw_pred,th_pred=preds[b,grid_y,grid_x,anchor_idx,:4]
                                x_c_pred=(grid_x+tx_pred)/13.0
                                y_c_pred=(grid_y+ty_pred)/13.0
                                w_pred=anchor_w*torch.exp(tw_pred)
                                h_pred=anchor_h*torch.exp(th_pred)
                                bbox_pred=[x_c_pred,y_c_pred,w_pred,h_pred]
                                #计算iou
                                iou=calculate_iou(bbox_pred,bbox_tgt)
                                iou_thresh=0.6#iou阈值
                                if iou>iou_thresh:#iou大于阈值(且类别预测准确)则判断预测准确
                                    total_correct+=1

    print(f"验证集总损失：{total_val_loss:.6f}")
    accuracy=total_correct/total_obj#验证集预测准确率
    print(f"预测准确率：{accuracy:.6f}")
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        best_epoch=i+1
        count=0
        torch.save(detector.state_dict(),f"./models/best_acc_model.pth")#保存准确率最佳的model
    else:
        count+=1
        if count>=patience:
            print("早停触发！")
            break

print("训练结束")
end_time=time.time()#训练计时结束
print(f"总用时：{end_time-start_time:.2f}s")
print(f"最佳训练轮次：{best_epoch}，最佳验证集准确率：{best_accuracy:.6f}")