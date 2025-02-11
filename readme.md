## RCAN-DPO
**This is a branch of modified RCAN[1] project with DPO strategy proposed in Alpha-LFM.**

## Dependencies
```
- python=3.8.8
- tensorflow-1.15.4+nv-cp38-cp38-win_amd64.whl
- easydict==1.9
- protobuf==3.20.3
- scipy==1.6.2
- scikit-image==0.18.1
- numpy==1.18.3
- matplotlib==3.4.1
```
## Directory Structure:
    └── model:
        contain RCAN model
    └── tensorlayer: 
        The third-party codes for building deep learning model (TensorFlow-based).
        Copyright (c) 2016~2020 The TensorLayer contributors. All rights reserved.
        License: Apache License
        Version: 1.8.1 
        URL: https://github.com/tensorlayer/TensorLayer
        Citation:
        @article{tensorlayer2017,
                author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
                journal = {ACM Multimedia},
                title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
                url     = {http://tensorlayer.org},
                year    = {2017}
            }
## Usage
### Network inference
We have provided example data (*'./example_data/lyso.tif'*) and trained weights (*'./checkpoint/Rab7_x4_[RCAN_multistage]'*) for users to quickly test the network.
* **Step 1:** 
<br> Set the parameters in 'config_test.py'. Detailed descriptions of the parameters are provided within the file as code annotations.
* **Step 2:** 
<br> Run *eval.py*. 
    ```
    python eval.py
    ```
### Network training
* **Step 1:** 
<br> Set the parameters in 'config.py'. Detailed descriptions of the parameters are provided within the file as code annotations.
* **Step 2:** 
<br> Run *train.py*
    ```
    python train.py
    ```
### Reference:
1. Image Super-Resolution Using Very Deep Residual Channel Attention Networks