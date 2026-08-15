
from ultralytics import YOLO
from val import BGD_YOLO
# 主函数
if __name__ == "__main__":

    # yolo_model = BGD_YOLO('ultralytics/models/v8/yolov8_wdw.yaml',detail_mode='true')
    yolo_model = YOLO('runs/train9/weights/last.pt')

    # yolo_model.predict(data="ultralytics/datasets/BG.yaml",source='/home/bme-2020/czy/mydata/images/test',batch=1,imgsz=640,workers=1,project='runs/detect',save_txt=True,iou=0.5)
    # yolo_model.val(data="ultralytics/datasets/BG.yaml",source='/home/bme-2020/czy/mydata/images/test',batch=1,imgsz=640,workers=1,project='runs/detect',save_txt=True,iou=0.5)
    yolo_model.val(data='ultralytics/datasets/BG.yaml',source='/home/bme-2020/czy/mydata/images/test',batch=1,imgsz=640,workers=1,project='runs/detect',save_txt=True,iou=0.5)

