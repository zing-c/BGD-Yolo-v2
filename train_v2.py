from dataLoader import DetailDetectionModel,joint_training
from ultralytics import YOLO
from val import BGD_YOLO

import gc
if __name__ == "__main__":



    # #
    stage1=False
    stage2=False

    if stage1 is True:
        yolo_model = BGD_YOLO( '/home/bme-2020/czy/BGD-Yolo/runs/yolo/train43/weights/best.pt',
                               detail_weight='run/detail_net_attn.pt',
                               model_weight_only_yolo=True,
                               target_layers=[-1],  # if detail_model adds in head  target_layers=[-1]  esle selects the layer in yaml
                               detail_mode=True,
                               conf_threshold=0.1,
                               detail_num=1,
                               head_select=['low'],
                               detail_position='head',
                        )

        torch_model = yolo_model.model

        # 定义你需要【训练】的模块名称关键字
        # 这些关键字必须能唯一匹配到你新加的那几层 (Conv, BN, ReLU, FC 等)
        trainable_keywords = [
            'conv_for_yolo_low',
            'bn_for_yolo_low',
            'relu_for_yolo_low',
            'conv_l.','conv_m.','conv_s.'
        ]

        trainable_count = 0
        frozen_count = 0
        for name, param in torch_model.named_parameters():
            # 1. 默认策略：先冻结所有参数 (包括 YOLO Backbone 和 Detail Net)
            param.requires_grad = False
            is_trainable = False

            # 2. 特例策略：如果参数名包含我们的关键字，则解冻
            for kw in trainable_keywords:
                if kw in name:
                    param.requires_grad = True
                    is_trainable = True
                    trainable_count += 1
                    break  # 只要匹配到一个关键字，就标记为可训练，跳出内层循环

            if not is_trainable:
                frozen_count += 1

        print(f">>> 冻结配置完成。")

        #  打印几个真正会参与训练的层名字，确保没有写错
        print(">>> 验证：以下是前 5 个将参与梯度的参数名称：")
        active_params = [n for n, p in torch_model.named_parameters() if p.requires_grad]
        for p_name in active_params[:5]:
            print(f"    [Active] {p_name}")

        # ==============================================================================
        # 3. 开始训练
        # ==============================================================================
        print(">>> 开始训练局部模块...")
        yolo_model.train(
            data="/home/bme-2020/czy/yolov10/data.yaml",
            lr0=0.01,
            batch=32,
            imgsz=640,
            epochs=30,
            project='yolo-runs/trainv2/',
            name='stage1_warmup'
        )



        print("\n>>> [Stage 1] 训练完成。内存中的模型权重已更新。")
    if stage2 is True:

    # # ==============================================================================
    # # 阶段 2：解冻所有层 & 全局微调 (Full Finetune Stage)
    # # ==============================================================================
    # print("\n>>> [Stage 2] 准备全网微调：解冻所有参数...")
    #
    # # 1. 解冻所有参数
    # for name, param in torch_model.named_parameters():
    #     param.requires_grad = True
    #
    # # 2. 验证解冻情况
    # frozen_check = [n for n, p in torch_model.named_parameters() if not p.requires_grad]
    # print(f"    - 剩余冻结参数数量: {len(frozen_check)} (应为0)")
    #
    #
    # # ... Stage 1 结束 ...
    #
    # print(">>> 清理 Stage 1 占用的内存...")
    # import gc
    #
    # gc.collect()  # 清理 Python 垃圾回收
    # torch.cuda.empty_cache()  # 清理 PyTorch 显存缓存
    #
    # # ==============================================================================
    # # 阶段 2 训练：全参数微调
    # # ==============================================================================
    # print("\n>>> [Stage 2] 开始全网微调 (Finetune)...")

    # 注意：这里直接复用 yolo_model，因为它保留了 Stage 1 训练后的权重
        yolo_model = BGD_YOLO('/home/bme-2020/czy/BGD-Yolo/yolo-runs/trainv2/stage1_warmup4/weights/best.pt',
                              # detail_weight='run/detail_net_attn.pt',
                              model_weight_only_yolo=False,
                              target_layers=[-1],
                              # if detail_model adds in head  target_layers=[-1]  esle selects the layer in yaml
                              detail_mode=True,
                              conf_threshold=0.1,
                              detail_num=1,
                              head_select=['low'],
                              detail_position='head',
                              )

        yolo_model.train(
                data = "/home/bme-2020/czy/yolov10/data.yaml",
                lr0 = 0.001,  # 【关键】降低学习率！通常是原来的 1/10，防止破坏预训练权重
                batch = 32,  # 如果显存不够（因为全解冻了），可能需要减小 batch
                imgsz = 640,
                epochs = 50,  # 进行更长时间的微调
                project='yolo-runs/trainv2/',
                name = 'stage2_finetune',  # 结果会保存在新文件夹
                warmup_epochs = 0  # 第二阶段通常不需要 warmup，或者设很短
            )



    #
    # # bgd_val

    yolo_model = BGD_YOLO('yolo-runs/trainv2/stage1_warmup4/weights/best.pt',
                        model_weight_only_yolo=False,
                        target_layers=[-1],
                        detail_mode=True,
                        conf_threshold=0.1,
                        detail_num=1,
                        head_select=['low'],
                        detail_position='head'
                        )
    yolo_model.val(data="/home/bme-2020/czy/yolov10/data.yaml",
                   split='test',imgsz=640, workers=1,batch=1,conf=0.5)