Installation <br>
=
MMaction2 depends on Pytorch,MMCV <br>
**Environment preparation<br>
conda create --name openmmlab python=3.8 -y<br>
conda activate openmmlab<br>
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch<br>
pip install -U openmim<br>
mim install mmengine<br>
mim install mmcv<br>
cd mmaction2<br>
pip install -v -e <br>
