```bash
apt install build-essential python3-dev libopenblas-dev
conda create -n sparse_gan python=3.8
conda activate sparse_gan

conda install openblas-devel -c anaconda
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/NVIDIA/MinkowskiEngine.git --depth 1 -b master
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
