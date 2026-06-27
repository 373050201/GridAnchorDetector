"""
模型测试
评估标准：平均精度均值mAP
"""
import time
import torch
import yaml
from yolo_dataset import YoloDataset
from torch.utils.data import DataLoader
from model import GridAnchorDetector
from calculate import calculate_mAP, nms_vectorized

print("正在加载数据集...")
batch_size=8
test_dataset=YoloDataset("test")
test_loader=DataLoader(test_dataset,batch_size=8,shuffle=True,drop_last=True)
with open("config.yaml",'r') as f:
    config=yaml.safe_load(f)
anchors=config["anchors"]
nc=config["nc"]
mAP_iou_threshold=config["mAP_iou_threshold"]

print("正在加载模型...")
detector=GridAnchorDetector()
detector=detector.cuda()
detector.load_state_dict(torch.load(f"models/best_mAP@{mAP_iou_threshold}_model.pth"))

print("开始测试...")
start_time=time.time()
detector.eval()#评估模式
all_preds=[]#所有预测框
all_tgts=[]#所有目标框
img_id=0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        preds = detector(imgs)
        # 解码
        for b in range(batch_size):
            init_preds = []  # 暂存nms前所有预测框
            for grid_y in range(13):
                for grid_x in range(13):
                    for anchor_idx in range(5):
                        anchor_w, anchor_h = anchors[anchor_idx]  # 锚框宽高

                        conf_tgt = labels[b, grid_y, grid_x, anchor_idx, 4]  # 目标框置信度解码
                        if conf_tgt > 0.5:
                            cls_tgt = torch.argmax(labels[b, grid_y, grid_x, anchor_idx, 5:])  # 目标框类别解码
                            # 目标框bbox解码
                            tx_tgt, ty_tgt, tw_tgt, th_tgt = labels[b, grid_y, grid_x, anchor_idx, :4]
                            x_c_tgt = (grid_x + tx_tgt) / 13.0
                            y_c_tgt = (grid_y + ty_tgt) / 13.0
                            w_tgt = anchor_w * torch.exp(tw_tgt)
                            h_tgt = anchor_h * torch.exp(th_tgt)
                            all_tgts.append([img_id, x_c_tgt, y_c_tgt, w_tgt, h_tgt, conf_tgt, cls_tgt])

                        conf_pred = preds[b, grid_y, grid_x, anchor_idx, 4]  # 预测框置信度解码
                        if conf_pred > 0.5:
                            cls_pred = torch.argmax(preds[b, grid_y, grid_x, anchor_idx, 5:])  # 预测框类别解码
                            # 预测框bbox解码
                            tx_pred, ty_pred, tw_pred, th_pred = preds[b, grid_y, grid_x, anchor_idx, :4]
                            x_c_pred = (grid_x + tx_pred) / 13.0
                            y_c_pred = (grid_y + ty_pred) / 13.0
                            w_pred = anchor_w * torch.exp(tw_pred)
                            h_pred = anchor_h * torch.exp(th_pred)
                            init_preds.append([x_c_pred, y_c_pred, w_pred, h_pred, conf_pred, cls_pred])  # 存入暂存区
            final_preds = nms_vectorized(init_preds)  # nms处理,去除重叠框
            final_preds = [[img_id] + sublist for sublist in final_preds]  # 在所有子列表第一个位置添加img_id
            all_preds.extend(final_preds)  # 将此图处理后的预测框加入所有预测框的列表
            img_id += 1
mAP=calculate_mAP(all_preds,all_tgts,nc,mAP_iou_threshold)#计算mAP
print(f"测试集mAP@{mAP_iou_threshold}：{mAP:.4f}")

print("测试结束")
end_time=time.time()
print(f"总用时：{end_time-start_time:.2f}s")