from dataLoader import DetailDetectionModel,joint_training
from ultralytics  import YOLO

if __name__ == "__main__":
    #
    # 训练v8
    # #
    # yolo_model = YOLO('ultralytics/models/v8/yolo_mid_low.yaml')
    # yolo_model.train(data="/home/bme-2020/czy/yolov10/data.yaml",imgsz=640,epochs=200,batch=32,project='/home/bme-2020/czy/BGD-Yolo/runs/yolo/')


    # yolo_model = YOLO('/home/bme-2020/czy/BGD-Yolo/runs/yolo/train10/weights/best.pt')
    yolo_model = YOLO('/home/bme-2020/czy/yolov8-20230616/yolov8/last.pt')
    yolo_model.model.eval()

    yolo_model.val(data="/home/bme-2020/czy/yolov10/data.yaml",imgsz=640,split='test', batch=1,conf=0.5,single_cls=True)