"""
模型测试
评估标准：预测准确率acc
"""
import time
import torch
import yaml
from yolo_dataset import YoloDataset
from torch.utils.data import DataLoader
from model import GridAnchorDetector
from calculate import calculate_iou



print("正在加载数据集...")
batch_size=8
test_dataset=YoloDataset("test")
test_loader=DataLoader(test_dataset,batch_size=8,shuffle=True,drop_last=True)
with open("config.yaml",'r') as f:
    config=yaml.safe_load(f)
anchors=config["anchors"]
nc=config["nc"]

print("正在加载模型...")
detector=GridAnchorDetector()
detector=detector.cuda()
detector.load_state_dict(torch.load("models/best_acc_model.pth"))

print("开始测试...")
start_time=time.time()
detector.eval()#评估模式
total_obj=0#总目标数(准确来说乘了锚框数)
total_correct=0#预测正确的锚框个数
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        preds = detector(imgs)
        # 解码
        for b in range(batch_size):
            for grid_y in range(13):
                for grid_x in range(13):
                    for anchor_idx in range(5):
                        if labels[b, grid_y, grid_x, anchor_idx, 4] > 0.5:
                            total_obj += 1
                            # 类别解码
                            cls_pred = torch.argmax(preds[b, grid_y, grid_x, anchor_idx, 5:])
                            cls_tgt = torch.argmax(labels[b, grid_y, grid_x, anchor_idx, 5:])
                            if cls_pred != cls_tgt:
                                continue
                            # 当前锚框宽高
                            anchor_w, anchor_h = anchors[anchor_idx]
                            # 目标框bbox解码
                            tx_tgt, ty_tgt, tw_tgt, th_tgt = labels[b, grid_y, grid_x, anchor_idx, :4]
                            x_c_tgt = (grid_x + tx_tgt) / 13.0
                            y_c_tgt = (grid_y + ty_tgt) / 13.0
                            w_c_tgt = anchor_w * torch.exp(tw_tgt)
                            h_c_tgt = anchor_h * torch.exp(th_tgt)
                            bbox_tgt = [x_c_tgt, y_c_tgt, w_c_tgt, h_c_tgt]
                            # 预测框bbox解码
                            tx_pred, ty_pred, tw_pred, th_pred = preds[b, grid_y, grid_x, anchor_idx, :4]
                            x_c_pred = (grid_x + tx_pred) / 13.0
                            y_c_pred = (grid_y + ty_pred) / 13.0
                            w_c_pred = anchor_w * torch.exp(tw_pred)
                            h_c_pred = anchor_h * torch.exp(th_pred)
                            bbox_pred = [x_c_pred, y_c_pred, w_c_pred, h_c_pred]
                            # 计算iou
                            iou = calculate_iou(bbox_tgt, bbox_pred)
                            iou_threshold = 0.6
                            if iou > iou_threshold:
                                total_correct += 1

accuracy=total_correct/total_obj
print(f"预测准确率：{accuracy:.6f}")
print("测试结束")
end_time=time.time()
print(f"总用时：{end_time-start_time:.2f}s")