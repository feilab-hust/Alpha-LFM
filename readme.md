
# Alpha-Net
**This is a branch of Alpha-LFM that only contains the Tensorflow implementation of neural network.**

# Requirements
- **System requirements**
```
· Windows 10. Linux should be able to run the code but the code has been only tested on Windows 10 so far.
· Python 3.8.8 
· CUDA 11.1 and cuDNN 8.2.0
· Graphics: NVIDIA RTX 3090, or better
· Memeory: > 128 GB 
· Hard Drive: ~50GB free space (SSD recommended)
```
- **Running environment requirements**
```
- python=3.8.8
- tensorflow-1.15.4+nv-cp38-cp38-win_amd64.whl
- easydict==1.9
- protobuf==3.20.3
- scipy==1.6.2
- scikit-image==0.18.1
- numpy==1.18.3
- matplotlib==3.4.1
- mat73==0.59
```
***Note: more details about dependencies installation can be found at the "User manual" in the main branch of Alpha-LFM***

# Directory Structure:
```    
├── Alpha-LFM

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
# Usage

### Model inference for quick validation
***Note: Users can quickly apply the [trained models](/checkpoint) to our provided [validation data](/example/validation_data)***.

* **Step 1:**
<br> Set the parameters in 'config_test.py'. For example, use the trained 'lysosome' model to conduct LFM 3D reconstruction
    ```
    label = 'lysosome_enhanced'
    validation_data_path= r'./example/validation_data/lyso'
    ```
* **Step 2:** Run 'eval.py' with the following command. 
<br> The results will be saved at the child folder "Recon_lysosome_enhanced". "0" in the command means the GPU ID.
    ```
    python eval.py -g 0
    ```
### Model training on paired data
* **Step 1:** 
<br> Set the parameters in 'config.py'. Detailed descriptions of the parameters are provided within the file as code annotations.
* **Step 2:** 
<br> Pre-train each sub-module in Alpha-Net (*e.g.*, denoising, de-aliasing, 3D Reconstruction). Users can type the following command in console:
    ```
    python ./pretrainers/preTrain_denoise.py -g 0
    python ./pretrainers/preTrain_ViewSR.py -g 1
    python ./pretrainers/preTrain_VCD.py -g 2
    ```
* **Step 3:** 
  <br> After the pre-training finished, user can train Alpha-net with the following command:
    ```
    python train.py -g 0
    ```
* **(*Optional*) Fine-tuning model with 2D WFs**
  <br>
  Set the parametes in 'config_finetune.py' and run the following commend
    ```
    python train_finetune.py -g 0
    ```
# Citation
If you use this code and relevant data, please cite the corresponding paper where original methods appeared: 
\
*Adaptive-learning physics-aware light-field microscopy enables day-long and millisecond-scale super-resolution imaging of 3D subcellular dynamics*

# Contact
Should you have any questions regarding this code and the corresponding results, please contact Lanxin Zhu (lanxinzhu@hust.edu.cn)

