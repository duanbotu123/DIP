# origin repo [sam2](https://github.com/facebookresearch/sam2)

# 想法

sam2用点确定要割出的物体，所以把pose估计估出的2d关键点输入，就可以直接得到人的mask

# run

按sam2官方的说明配置完环境，下载预训练模型，运行scripts/image_predict.py

或者运行脚本scripts/get_mask.sh，需要改下文件路径

- args
    - image: 图片的路径
    - annots: 2d关键点路径
    - mask: 保存mask的路径
    - (optional) checkpoint: 用到的模型
    - (optional) model_cfg: 配置文件

# todo 

sam2有图片模式和视频模式，看一下视频模式怎么用