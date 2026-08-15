import wandb
from ultralytics import YOLO
from val import BGD_YOLO

if __name__ == "__main__":
    # --- 1. 统一配置超参数 ---
    # 将训练参数和自定义的模型参数整合在一个字典中
    #exp3_4_1.pt   数据集加入负样本
    hyp_params = {
        "data": "/home/bme-2020/czy/yolov10/data.yaml",
        "lr0": 0.001,
        "batch": 32,
        "imgsz": 640,
        "epochs": 40,
        "workers": 0,
        "detail_conf_threshold": 0.5,
        "detail_weight":'run/detail_net_atten/exp3_4_1.pt',
        "yolo_weight":'/home/bme-2020/czy/BGD-Yolo_v2/runs/yolo/train10/weights/best.pt',
        "project_path": "yolo-runs/bgd",
        "experiment_name": "bgd(detail_conf=0.5)"
    }

    # --- 2. 初始化 wandb ---
    # 这样 wandb 会自动保存 hyp_params 里的所有内容
    wandb.init(
        project="BGD-YOLO",
        name=hyp_params["experiment_name"],
        config=hyp_params
    )

    # --- 3. 初始化模型 ---
    # 使用 hyp_params 里的变量
    yolo_model = BGD_YOLO(
        model=hyp_params["yolo_weight"],
        detail_weight=hyp_params["detail_weight"],
        model_weight_only_yolo=True,
        target_layers=[-1],
        detail_mode=True,
        conf_threshold=hyp_params["detail_conf_threshold"],
        detail_num=1,
        head_select=['low'],
        detail_position='head',
    )

    # --- 4. 运行训练 ---
    # 通过 ** 语法将字典中匹配的键值对自动传给 train 函数
    yolo_model.train(
        data=hyp_params["data"],
        lr0=hyp_params["lr0"],
        batch=hyp_params["batch"],
        imgsz=hyp_params["imgsz"],
        epochs=hyp_params["epochs"],
        workers=hyp_params["workers"],
        project=hyp_params["project_path"],
        name=hyp_params["experiment_name"],

    )

    # --- 5. 结束任务 ---
    wandb.finish()