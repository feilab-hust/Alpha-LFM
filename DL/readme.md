
#### The deep-learning related codes for Alpha-LFM
***
Directory Structure:
```    
├── DL

    └── model:
        Different deep learning models, including LF denoising model, View SR model, and 3D reconstruction model.
        └── util:
            Functions in various DL models.
            
    └── pretrainers: 
        Functions for pre-training the LF-denoising network, View-SR network, and 3D reconstruction network.
        
    └── misc: 
        Functions for data loading and processing.
        
    └── logs (generated only when network training):
        Folder for saving logs file during network training 
        └── samples:
            The sampled images during network training.
        └── tensorboard:
            Tensorboard files stored the loss plots when training and network graph.

    └── checkpoint:
        Folder contains the model weights of trained network.
        Note: For fast implementation alpha-LFM, we have provided the trained models (e.g. lysosome_enhanced, mito2matrix_finetuning and mito_enhanced)

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
```
