# -*- coding: utf-8 -*-
import jittor as jt
from Process import Trainer
from MakeTxt import create_dataset_txt_files
import argparse

'''
---------------先使用DataAugmentation进行数据的填充---------------
先运行DataAugmentation进行弹性形变数据增强
参数:
--img_path "C:/Users/24174/Desktop/cmmSecond/Unet-jittor/data/ISBC2012/train/imgs" 
--mask_path "C:/Users/24174/Desktop/cmmSecond/Unet-jittor/data/ISBC2012/train/labels" 
--sum 2 
--alpha_lv 2 
--sigma_lv 0.08 
--alpha_affine_lv 0.08
'''

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='process confing')

    parser.add_argument('--Data_directory', type=str, default='./data/ISBC2012',
                        help='Data directory ')

    parser.add_argument('--Datatxt_directory', type=str, default='./DataTxt_ISBS2012',
                        help='DataTxt_ISBS2012 directory ')

    parser.add_argument('--Log_directory', type=str, default='./logs_ISBS2012',
                        help='Log directory ')

    parser.add_argument('--Checkpoint_directory', type=str, default='./checkpoints_ISBS2012',
                        help='Checkpoint directory ')

    parser.add_argument('--Result_directory', type=str, default='./results_ISBS2012',
                        help='Result directory ')

    parser.add_argument('--is_grayscale', type=str, default=False,
                        help='is_grayscale')

    parser.add_argument('--n_channels', type=int, default=1,
                        help='n_channels')

    parser.add_argument('--n_classes', type=int, default=2,
                        help='n_classes')

    parser.add_argument('--Total_epochs', type=int, default=400,
                        help='Total_epochs')

    parser.add_argument('--Val_epochs', type=int, default=200,
                        help='Val_epochs')

    parser.add_argument('--Batch_size', type=int, default=1,
                        help='Batch_size')

    parser.add_argument('--dropout_probs', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5],
                       help='List of dropout probabilities (0.2 0.3 0.4 0.5)')

    parser.add_argument('--bilinear', type=str, default=False,
                        help='bilinear')

    parser.add_argument('--Learning_rate', type=float, default=0.001,
                        help='Learning_rate')

    parser.add_argument('--Loss_w0', type=int, default=10,
                        help='Loss_w0')

    parser.add_argument('--Loss_sigma', type=float, default=512*0.08,
                        help='Loss_sigma')

    parser.add_argument('--mode', type=str, default='Unet',
                        help='mode')

    args = parser.parse_args()

    '''
    参数设置
    --Data_directory "./data/ISBC2012"     #Data文件夹    注意 image和label都应是512*512,因为我用的卷积是Unet论文中的valid卷积(无填充卷积),所以要注意尺寸匹配问题
    --Datatxt_directory "./DataTxt"        #存放Data中image和label路径的DataTxt文件夹 没有会自动创建
    --Log_directory "./logs"               #记录log的文件夹 没有会自动创建
    --Checkpoint_directory "./checkpoints" #保存模型的文件夹 没有会自动创建
    --Result_directory "./results"         #保存最优结果的文件夹 没有会自动创建
    --is_grayscale True                    #image是不是灰度(黑白图片) 如果是1个通道的话就是 True 如果是3个通道就是False
    --n_channels 1                         #模型输入通道数
    --n_classes 2                          #模型输出通道数
    --Total_epochs 100                     #总共的epoch
    --Val_epochs 50                        #从第几轮epoch开始进行验证
    --Batch_size 1                         #Batch_size大小 我是3060所以设为1(设为1的GPU占用率也是99%)
    --bilinear False                       #Unet 模型的上采样的方式选择 True-双线性插值进行上采样(速度快) Flase-使用转置卷积上采样(速度满,有可训练参数)
    --dropout_probs 0.1 0.2 0.3 0.4        #4个编码器(下采样)尾端的dropout率
    --Learning_rate 0.0001                 #学习率
    --Loss_w0 10                           #损失函数中的 计算权重图时用的参数w0    具体可看论文中的公式
    --Loss_sigma 5   512*0.08=40.96        #损失函数中的 计算权重图时用的参数sigma 具体可看论文中的公式
    --mode "Unet" 或 "U2netS" 或 "U2net"          #模型选择
    '''
    #创建保存image label路径的txt文件
    create_dataset_txt_files(args.Data_directory, args.Datatxt_directory)

    # 检查GPU是否可用
    jt.flags.use_cuda = 1
    device = "cuda" if jt.flags.use_cuda else "cpu"
    print(f"使用设备: {device}")

    # 优化性能
    jt.flags.log_silent = 1  # 关闭日志输出

    #创建Train类
    trainer = Trainer(
        data_dir=args.Datatxt_directory,
        log_dir=args.Log_directory,
        checkpoint_dir=args.Checkpoint_directory,
        result_dir=args.Result_directory,
        is_grayscale=(args.is_grayscale.lower() == 'true'),
        mode=args.mode,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        total_epochs=args.Total_epochs,
        val_start_epoch=args.Val_epochs,
        batch_size=args.Batch_size,
        dropout_probs=args.dropout_probs,
        bilinear=(args.bilinear.lower() == 'true'),
        learning_rate=args.Learning_rate,
        device=device,
        w0=args.Loss_w0,
        sigma=args.Loss_sigma,
    )
    # 开始训练
    trainer.run()
