# HOI_Gen


## Code Structure

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
├── human_recon/                 #  mask+video+smpl_finetune->3dgs
├── object_tracking/           #  mask+video->...
├──  ...              
└──  README.md                       
