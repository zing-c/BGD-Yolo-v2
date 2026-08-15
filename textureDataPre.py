from ultralytics import YOLO
import os
import sys
# 加载预训练的 YOLOv8 模型
yolo_model = YOLO('yolov8n.pt')
yolo_model=YOLO('runs/detect/train27/weights/best.pt')

dataset={ 'train':"E:/cv/czy/mydata/images/train" ,
          'test':"E:/cv/czy/mydata/images/test" ,
          'val':"E:/cv/czy/mydata/images/val" }



# image_files{'train':[image_file]，'test':[image_file···]，'val':[image_file···]，}
image_files ={}
for key,value in dataset.items():
    image_files[key] = [os.path.join(value, f) for f in os.listdir(value) if f.endswith(('.jpg', '.png', '.jpeg'))]


demo1={}
demo1['train']=image_files['train']
bbox={}
for key,value in demo1.items():
    for i in value:
        results = yolo_model.predict(i)
        # 提取检测结果
        for result in results:
            boxes = result.boxes  # 获取边界框信息
            for box in boxes:
                # 提取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox[i]=[x1, y1, x2, y2]
                if(len(bbox)==15):
                    print(bbox)
                    sys.exit(0)

                # 提取类别标签
                class_id = box.cls[0].item()  # 类别 ID
                label = yolo_model.names[class_id]  # 类别名称

                # # 打印结果
                # print(f"Image: {image_path}")
                # print(f"Box: [{x1}, {y1}, {x2}, {y2}], Label: {label}")
                # print("-" * 40)

