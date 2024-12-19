# HOI_Gen

## requirement
conda create -n hoi_g python=3.10

cd to root dir

## data download 
wget 


## video to images
pip install opencv-python

python ./pre_processing/data_proc/video2img.py

## easyMocap 2d detection

if no nvcc, download&install cuda toolkit 12.1

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

python -m pip install -r ./human_tracking/EasyMocap/requirements.txt

cd ./human_tracking/EasyMocap

python setup.py develop


## sam2

cd checkpoints && \
./download_ckpts.sh && \
cd ..


## human_tracking



## Input-Output

general input to output : 
- object videos + interaction videos + human videos->human gaussian sequence  + object gaussian sequence


detailed pipeline:
```
object_videos -> mask -> mesh ->static object gaussian
                                         │
interaction_videos -> mask               │
                         └──> object tracking ->animated object gaussian sequence
                         │       |                                  │
                         │       |                                  │
                         └──> smplx                                 │
                                 └──> smplx_finetune                │
                                                │                   │
humen_videos -> mask -> static human gaussian   │                   │
                              └────────┬────────┘                   │
                                       │                            │
                        animated human gaussian sequence            │
                                       └──────────────┬─────────────┘
                                                      └──> HOI gaussian sequence

```


## Code Structures

```
root/
├── data/                     # video  （input->output）
│   ├── env/                 #room environment
│   │   ├── 001_0001/      
│   │   ├── ...           
│   │   └── 016_0001/           
│   ├── human/            #human videos
│   ├── interaction/              #human-object interaction videos
│   ├── object/             # object images
│   └── delete.py
├── pre_processing/                   # ...
│   ├── ...               
│   └── output_.../                 
├── human_tracking/                     #  ...
│   ├── ...               
│   └── output_.../
├── first_frame/                     #  ...
│   ├── ...               
│   └── output_.../
├── object_tracking/           #  ...
│   ├── ...               
│   └── output_.../ 
├── pose_finetune/               #  object_mesh+smplx->smplx_finetune
│   ├── codes              
│   ├── ...
│   └── output_smplx_finetune   # finetune后的smplx输出目录
├── human_recon/                 #  ...
│   ├── ...               
│   └── output_.../ 
├── object_recon/                    # ...
│   ├── ...               
│   └── output_.../ 
├──  ...              
└──  README.md                       
```

## visualization of final output

![HOI gaussians](path/to/your/gif)

