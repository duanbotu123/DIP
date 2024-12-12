# HOI_Gen


## Input-Output

general input to output : 
- videos->human gaussian sequence  + object gaussian sequence


detailed pipeline:
```                                
object_videos -> mask -> mesh -> object gaussian
                                                │
interaction_videos -> mask                      │
                         └──> smplx , object tracking ->object gaussian sequence
                            └──> smplx_finetune                 │
                                        │                       |
humen_videos -> mask -> static human gaussian                   |
                                        └──>animated human gaussian sequence    
                                                                └──>HOI gaussian sequence
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
├── pre_processing/                   # video->mask
│   ├── ...               
│   └── ...                 
├── object_recon/                    # mask+video->3dgs
├── human_tracking/                     #  mask+video->smpl
├── pose_finetune/               #  smpl->smpl_finetune
│   ├── codes              
│   ├── ...
│   └── output_smplx   # output smplx files
├── human_recon/                 #  mask+video+smpl_finetune->3dgs
├── object_tracking/           #  mask+video->...
├──  ...              
└──  README.md                       
```

## visualization of every steps

![描述](path/to/your/gif)

