# HOI_Gen

## 协作
数据集上传&下载
```
wget 
```


- 更改pipeline细节

- 更改code structure：
例子


- readme 例子
/pose_finetune

使用：
python main.py

输入：


输出：


可视化：



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

