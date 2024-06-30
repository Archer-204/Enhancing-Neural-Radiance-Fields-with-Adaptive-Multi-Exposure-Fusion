## Enviroment setup:

We build the environment based on  [nerf-factory](https://github.com/kakaobrain/nerf-factory) and [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF):

Python: 3.8 , PyTorch: 1.11.0 , Cuda: 11.3
```
$ git clone https://github.com/Archer-204/AME-NeRF.git
$ cd AME-NeRF
$ conda create -n ame_nerf -c anaconda python=3.8
$ conda activate aleth_nerf
$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## LOM dataset:

Follow the instructions to download the LOM dataset in [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF)

## Testing:

```
# bike
$python ./source/main.py --config ./source/configs/lom.yaml --pipeline test --checkpoint ./source/ckpt/last.ckpt && python run.py --ginc configs/LOM/nerf/nerf_bike.gin --logbase ./logs --ginb run.run_train=False 
```


