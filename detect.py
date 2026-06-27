"""
模型部署
"""
import torch
from PIL import Image
from model import GridAnchorDetector
from torchvision import transforms
from calculate import nms,visualize_bbox
import yaml



def load_img(path):#加载模型输入图片
    img=Image.open(path)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416,416))
    ])
    img=transform(img)
    return img



print("正在加载配置...")
conf_threshold=0.5#若conf大于此值,判断为正样本
iou_threshold=0.5#若iou大于此值,判断为重叠框
model_path="models/best_mAP@0.5_model.pth"
img_path="detect/detect.jpg"
with open("config.yaml",'r') as f:
    config=yaml.safe_load(f)
anchors=config["anchors"]

print("正在加载模型...")
detector=GridAnchorDetector()
detector=detector.cuda()
detector.load_state_dict(torch.load(model_path))
detector.eval()

print("正在预测...")
img=load_img(img_path)
img.unsqueeze_(0)
img=img.cuda()
pred=detector(img)
pred=pred[0]#shape:[13,13,5,5+C]
img=Image.open(img_path)#原始图片

bboxes=[]
for grid_y in range(13):
    for grid_x in range(13):
        for anchor_idx in range(5):
            anchor_w,anchor_h=anchors[anchor_idx]
            conf=pred[grid_y,grid_x,anchor_idx,4]
            if conf>conf_threshold:#若conf>conf_threshold,判断为正样本
                #预测框解码
                tx,ty,tw,th=pred[grid_y,grid_x,anchor_idx,:4]
                cls=torch.argmax(pred[grid_y,grid_x,anchor_idx,5:])#类别
                x_c=(grid_x+tx)/13.0#bbox
                y_c=(grid_y+ty)/13.0
                w=anchor_w*torch.exp(tw)
                h=anchor_h*torch.exp(th)
                bboxes.append([x_c,y_c,w,h,conf,cls])

print(f"nms前bbox总数：{len(bboxes)}")
bboxes=nms(bboxes,iou_threshold)#非极大值抑制,去除重叠框
print(f"nms后bbox总数：{len(bboxes)}")
visualize_bbox(img,bboxes)#可视化预测框
img.save("detect/result.jpg")