
from ultralytics import YOLO
from val import BGD_YOLO

if __name__ == "__main__":
    #
    # 训练v8
    #
    # yolo_model = YOLO()
    # yolo_model.train(data="/home/bme-2020/czy/yolov10/data.yaml", batch=1, imgsz=640,epochs=300)

    # # val
    #  yolo_model = YOLO('/home/bme-2020/czy/BGD-Yolo/runs/yolo/train/weights/best.pt',)
    #  yolo_model.val(data="/home/bme-2020/czy/yolov10/data.yaml", split='test', imgsz=640, workers=1, batch=1)
    #

    #训练v8

    # yolo_model = BGD_YOLO('/home/bme-2020/czy/BGD-Yolo/runs/yolo/train/weights/best.pt',model_weight_only_yolo=True,detail_weight=None,target_layers=[-1],
    #                           detail_mode=True)
    # yolo_model.train(data="/home/bme-2020/czy/yolov10/data.yaml", batch=32, imgsz=640, epochs=300,
    #                project='yolo-runs/train')
    #

    # bgd_val
  # yolo_model = BGD_YOLO('yolo-runs/train/train332/weights/last.pt',model_weight_only_yolo=False,head_select='low',
  #                       target_layers=[-1],detail_mode=True,conf_threshold=0.5)
  # yolo_model.val(data="/home/bme-2020/czy/yolov10/data.yaml", split='test',imgsz=640, workers=1,batch=1,conf=0.25)
  #

    #exp3_4_2.pt  去掉 max_pooling 改两个 fc + 数据集加入负样本
    #exp3_4_2.pt  去掉 max_pooling 改1个 fc + 数据集加入负样本
    #exp3_4_1.pt   数据集加入负样本
    #exp3_3_1.pt detail_model_atten 原模型
    yolo_model = BGD_YOLO( '/home/bme-2020/czy/BGD-Yolo_v2/runs/yolo/train10/weights/best.pt',
                           detail_weight='run/detail_net_atten/exp3_4_1.pt',#run/detail_net_atten.pt
                           model_weight_only_yolo=True,
                           target_layers=[-1],  # if detail_model adds in head  target_layers=[-1]  esle selects the layer in yaml
                           detail_mode=True,
                           conf_threshold=0.5,
                           detail_num=1,
                           head_select=['low'],
                           detail_position='head',


                           )

    yolo_model.train(
        data="/home/bme-2020/czy/yolov10/data.yaml",
        lr0=0.001,
        batch=32,
        imgsz=640,
        epochs=1,
        workers=0,
        cache=False,
        project="yolo-runs/train",
        name="bgd(detail_conf=0.5)"

    )




    # # bgd_val  run/detail_net_atten/exp2_3_1.pt
    #/home/bme-2020/czy/BGD-Yolo_v2/runs/yolo/train10/weights/best.pt
    # yolo_model = BGD_YOLO('/home/bme-2020/czy/BGD-Yolo_v2/yolo-runs/train/train604/weights/best.pt',
    #                       # detail_weight='run/detail_net_atten/exp3_3_1.pt',
    #                     model_weight_only_yolo=False,
    #                     target_layers=[-1],
    #                     detail_mode=True,
    #                     conf_threshold=0.5,
    #                     detail_num=1,
    #                     head_select=['low'],
    #                     detail_position='head'
    #                     )
    # yolo_model.val(data="/home/bme-2020/czy/yolov10/data.yaml",
    #                split='test',imgsz=640, workers=1,batch=1,conf=0.5)
