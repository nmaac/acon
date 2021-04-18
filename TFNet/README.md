# [TFNet](https://arxiv.org/pdf/2009.04759.pdf)
This repository contains TFNet implementation by Pytorch.


### TFNet
To show the effectiveness of the proposed acon family, we provide an extreme simple toy funnel network (TFNet) made only by pointwise convolution and ACON-FReLU operators.

<img src="https://user-images.githubusercontent.com/5032208/113963614-7a3a8200-985c-11eb-8946-65c0bcef0a80.png"  width=60%>


## Main results



The simple TFNet without the SE modules can outperform the state-of-the art light-weight networks without the SE modules.

|                   | FLOPs | #Params. |   top-1 err.   |
|-----------------  |:-----:|:--------:|:--------------:|
|  MobileNetV2 0.17 |  42M  |   1.4M   |    52.6    |
| ShuffleNetV2 0.5x |  41M  |   1.4M   |    39.4    |
|     TFNet 0.5     |  43M  |   1.3M   |  **36.6 (+2.8)**  |
|  MobileNetV2 0.6  |  141M |   2.2M   |    33.3    |
| ShuffleNetV2 1.0x |  146M |   2.3M   |    30.6    |
|     TFNet 1.0     |  135M |   1.9M   |  **29.7 (+0.9)**  |
|  MobileNetV2 1.0  |  300M |   3.4M   |    28.0    |
| ShuffleNetV2 1.5x |  299M |   3.5M   |    27.4    |
|     TFNet 1.5     |  279M |   2.7M   |  **26.0 (+1.4)**  |
|  MobileNetV2 1.4  |  585M |   5.5M   |    25.3    |
| ShuffleNetV2 2.0x |  591M |   7.4M   |    25.0    |
| TFNet 2.0         |  474M |   3.8M   |  **24.3 (+0.7)**  |




## Trained Models
- OneDrive download: [Link](https://1drv.ms/u/s!AgaP37NGYuEXhWbwpi4SX1IX6gOs?e=wIQYs1)
- BaiduYun download: [Link](https://pan.baidu.com/s/18uDVWe-rh4b7qI_NBvWUCw) (extract code: 13fu)


## Usage

### Requirements
Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh


Train:
```shell
python train.py  --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```
Eval:
```shell
python train.py --eval --eval-resume YOUR_WEIGHT_PATH --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```


## Citation
If you use these models in your research, please cite:

    @inproceedings{ma2021activate,
      title={Activate or Not: Learning Customized Activation},
      author={Ma, Ningning and Zhang, Xiangyu and Liu, Ming and Sun, Jian},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      year={2021}
    }
