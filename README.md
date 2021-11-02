# Chair Classification by function
This code was tested with python 3.7.

### Install dependencies

```
python -m pip install -r requirements.txt
```  

###  Train
This script is based on 4 type function of chair dataset as an example. For training, please run:

```
python TestRun.py
```

## Script Introduction

```SE_Block.py``` is a channel-wise attention that used to select most important feature map by Squeeze-and-Excitation [J. Hu, CVPR'18].

```ResNet.py``` is the Deep Residual Network that use residual learning to solve the vanishing gradient problem at deep neural network [K. He, CVPR'16].

```SENet.py``` follow the sequeeze and excitation,  "capture features in the convolution",  to make the network more efficient.

```SP&A-Net-Test-Run.ipynb``` is in the form of a Jupyter Notebook as a simple display with chair dataset as the training object.
