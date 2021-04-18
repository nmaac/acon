
## CVPR 2021 | Activate or Not: Learning Customized Activation.

This repository contains the official Pytorch implementation of the paper [Activate or Not: Learning Customized Activation, CVPR 2021](https://arxiv.org/pdf/2009.04759.pdf).

### ACON

We propose a novel activation function we term the ACON that explicitly learns to activate the neurons or not. 
Below we show the ACON activation function and its first derivatives. β controls how fast the first derivative asymptotes to the upper/lower bounds, which are determined by p1 and p2.


<img src="https://user-images.githubusercontent.com/5032208/113257297-fc76f380-92fc-11eb-9559-39d033baea4c.png" width=90%>

<img src="https://user-images.githubusercontent.com/5032208/113257194-cfc2dc00-92fc-11eb-94a0-f81569bed15e.png" width=90%>

### Training curves
We show the training curves of different activations here.

<img src="https://user-images.githubusercontent.com/5032208/113260052-65ac3600-9300-11eb-8d2f-ef968be1c3a2.png"  width=60%>


### TFNet
To show the effectiveness of the proposed acon family, we also provide an extreme simple toy funnel network (TFNet) made only by pointwise convolution and ACON-FReLU operators.

<img src="https://user-images.githubusercontent.com/5032208/113963614-7a3a8200-985c-11eb-8946-65c0bcef0a80.png"  width=60%>




## Main results

The following results are the ImageNet top-1 accuracy relative improvements compared with the ReLU baselines. The relative improvements of Meta-ACON are about twice as much as SENet.

<img src="https://user-images.githubusercontent.com/5032208/113256618-fcc2bf00-92fb-11eb-9b1d-8f0589009a9b.png" width=60%>

The comparison between ReLU, Swish and ACON-C. We show improvements without additional amount of FLOPs and parameters:
| Model             | FLOPs | #Params. | top-1 err. (ReLU) | top-1 err. (Swish) |   top-1 err. (ACON)   |
|-------------------|:-----:|:--------:|:-----------------:|:------------------:|:---------------------:|
| ShuffleNetV2 0.5x |  41M  |   1.4M   |        39.4       |     38.3 (+1.1)    |    **37.0 (+2.4)**    |
| ShuffleNetV2 1.5x |  299M |   3.5M   |        27.4       |     26.8 (+0.6)    |    **26.5 (+0.9)**    |
| ResNet 50         |  3.9G |   25.5M  |        24.0       |     23.5 (+0.5)    |    **23.2 (+0.8)**    |
| ResNet 101        |  7.6G |   44.4M  |        22.8       |     22.7 (+0.1)    |    **21.8 (+1.0)**    |
| ResNet 152        | 11.3G |   60.0M  |        22.3       |     22.2 (+0.1)    |    **21.2 (+1.1)**    |


Next, by adding a negligible amount of FLOPs and parameters, meta-ACON shows sigificant improvements:
| Model                         | FLOPs | #Params. |        top-1 err.      | 
|-------------------------------|:-----:|:--------:|:----------------------:|
| ShuffleNetV2 0.5x (meta-acon) | 41M   | 1.7M     |   **34.8 (+4.6)**      | 
| ShuffleNetV2 1.5x (meta-acon) | 299M  | 3.9M     |   **24.7 (+2.7)**      | 
| ResNet 50 (meta-acon)         | 3.9G  | 25.7M    |   **22.0 (+2.0)**      | 
| ResNet 101 (meta-acon)        | 7.6G  | 44.8M    |   **21.0 (+1.8)**      | 
| ResNet 152 (meta-acon)        | 11.3G | 60.5M    |   **20.5 (+1.8)**      | 





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

