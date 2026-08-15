from dataLoader import DetailDetectionModel, joint_training
from ultralytics_copy import YOLO
from val import BGD_YOLO

if __name__ == "__main__":
    #
    # 训练v8
    #
    # yolo_model = YOLO('ultralytics/models/v8/yolov8.yaml')
    # yolo_model.train(data="/home/bme-2020/czy/yolov10/data.yaml", batch=32, imgsz=640,epochs=300)
    #

    # 训练v8
    # #
    yolo_model = BGD_YOLO('ultralytics/models/v8/yolov8.yaml',detail_weight=None,target_layers=[2],
                              detail_mode=True)
    yolo_model.train(data="/home/bme-2020/czy/yolov10/data.yaml", batch=32, imgsz=640, epochs=300,
                   project='yolo-runs/train')



