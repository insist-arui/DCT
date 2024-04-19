**Installation
MMaction2 depends on Pytorch,MMCV
**Environment preparation
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmengine
mim install mmcv
cd mmaction2
pip install -v -e 
