
# Alpha-Net
**This is a branch of Alpha-LFM that contains only the source code related to the neural network.**

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

# Usage

### Model inference for quick validation
***Note: Users can fast implement the [trained models](/checkpoint) on our provided [validation data](/example/validation_data)***.
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
    python ./pretrainers/preTrain_ViewSR.py -g 2
    python ./pretrainers/preTrain_VCD.py -g 3
    ```


# Citation
If you use this code and relevant data, please cite the corresponding paper where original methods appeared: 
\
Sustained 3D super-resolution imaging of subcellular dynamics using adaptive-learning physics-aware light-field microscopy. 

# Contact
Correspondence Should you have any questions regarding this code and the corresponding results, please contact Lanxin Zhu (lanxinzhu@hust.edu.cn)

