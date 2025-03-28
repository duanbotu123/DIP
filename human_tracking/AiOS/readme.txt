# env: aios_121
conda create -n aios_121 python=3.8
conda activate aios_121
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
cd pytorch3d
pip install .
# Install MMCV, build from source
git clone -b v1.6.1 https://github.com/open-mmlab/mmcv.git
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_MLU=1
pip install -v -e .
cd ..
conda install -c conda-forge ffmpeg
pip install -r requirements.txt 
# Build deformable detr
cd models/aios/ops
python setup.py build install
cd ../../..
pip install icecream

# env: aios
conda create -n aios python=3.8
conda activate aios
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
cd pytorch3d
pip install .
# Install MMCV, build from source
git clone -b v1.6.1 https://github.com/open-mmlab/mmcv.git
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_MLU=1
pip install -v -e .
cd ..
conda install -c conda-forge ffmpeg
pip install -r requirements-xj.txt 
# Build deformable detr
cd models/aios/ops
python setup.py build install
cd ../../..
pip install icecream