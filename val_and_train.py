from dataLoader import DetailDetectionModel,joint_training
from ultralytics import YOLO
from val import BGD_YOLO

if __name__ == "__main__":

    #训练v8
    # yolo_model = YOLO('ultralytics/models/v8/yolov8.yaml',target_layers=[18],detail_weight='run/detail_net_attn.pt',
    #                           detail_mode=True)
    yolo_model = YOLO('ultralytics/models/v8/yolov8.yaml')
    yolo_model.train(data="/home/bme-2020/czy/yolov10/data.yaml", batch=32, imgsz=640,epochs=200,project= 'yolo-runs/train')

