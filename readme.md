
# Alpha-Net
**This is a branch of Alpha-LFM that contains only the source code related to the neural network.**

# Contents
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Citation](#Citation)
- [Contact](#Contact)
- [ToDo](#ToDo)
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
**Note: more details about dependencies installation can be found at the "User manual" in the main branch of Alpha-LFM**

# Usage

### Model inference for quick validation
   * Users can fast implement the [trained models](/checkpoint) on our provided [validation data](/example/validation_data).
     The step-by-step illustration of this process are detailed in Section 3.3, "Quick start guide",[Alpha-LFM manual](/Alpha-LFM%20Manual.pdf). The reconstruction results of example data are available at [Google Drive](https://drive.google.com/file/d/12gmTYRYSmYBnTj5Ha90yrMZVrBdFwLsv/view?usp=sharing).

### Model training on paired data

# Citation
If you use this code and relevant data, please cite the corresponding paper where original methods appeared: 
\
Sustained 3D super-resolution imaging of subcellular dynamics using adaptive-learning physics-aware light-field microscopy. 

# Contact
Correspondence Should you have any questions regarding this code and the corresponding results, please contact Lanxin Zhu (lanxinzhu@hust.edu.cn)

# ToDo:
- Upload more training example
- Rewrite the Python data-generation module to enhance its usability across a broader range of applications 
