# TTRD3:Texture Transfer Residual Denoising Dual Diffusion Model
 
## Abstract 
Remote Sensing Image Super-Resolution (RSISR) aims to reconstruct high-resolution (HR) remote sensing (RS) images from low-resolution (LR) inputs, overcoming physical limitations of imaging systems. As a specialized RS domain task, RSISR supports downstream fine-grained ground object interpretation.
 
Traditional RSISR methods face three key challenges:
1. Difficulty extracting effective multi-scale features from complex wide-area RS images 
2. Insufficient prior information leading to poor semantic consistency 
3. Imbalance between geometric accuracy and visual quality 
 
Our proposed Texture Transfer Residual Denoising Dual Diffusion Model (TTRD3) addresses these through:
- **MFAB**: Multi-scale Feature Aggregation Block for heterogeneous feature extraction 
- **STTG**: Sparse Texture Transfer Guidance for reference image utilization 
- **RDDM**: Residual Denoising Dual Diffusion Model framework for optimization 
---

## Installation 
- Python 3.10 on Ubuntu 22.04
- CUDA 11.8 and gcc 11.4 and corresponding supported pyTorch
- Python packages:
```
conda create -n TTRD3 python=3.10
conda activate TTRD3
pip install -r ./codes/requirements.txt
```

## Pretrained Weights 
- **Step I.** download weights:
If you need pre-trained weights, please download them from [Baidu Netdisk Link](https://pan.baidu.com/s/1qWyPJBRn72eojwucDJAhgw?pwd=jxb3). We provide three pre-trained weights, These correspond to three network models of different scales:  
  - `TTRD3-1Net-A`  
  - `TTRD3-1Net-B`  
  - `TTRD3-2Net`
- **Step II.** Place the pre-trained weights into the path TTRD3-main/codes/example/TTRD3/results/sample.


## Architecture Selection 
To use different residual denoising dual-diffusion model architectures, modify the `share_encoder` parameter in `TTRD3.yml`:
- `share_encoder = -1` → `TTRD3-1Net-A` architecture
- `share_encoder = 1` → `TTRD3-1Net-B` architecture
- `share_encoder = 0` → `TTRD3-2Net` architecture  

## Dataset Processing Guide 
 
### Overview 
This guide explains how to process input images to generate High-Resolution (HR) and Bicubic Super-Resolution (SR) results using the provided preprocessing script. Follow these steps to prepare your data for the TTRD3 framework.


Execute the preprocessing script with the following command:  
```bash 
python TTRD3-main/codes/datasets/data_preprocessing.py  \
  --path /path/to/your/input_dir \
  --out /path/to/processed_data  
```


---
## Train
- **Step I.** Enter directory:
```
cd ./codes/example/TTRD3
```
- **Step II.** Modify the dataroots for train and val based on your actual root in
  - `./codes/example/TTRD3/options/TTRD3.yml`
- **Step III.** Start training:
```
sh train.sh
```
- **Optional.** Also, you can change other training configurations in `TTRD3.yml` for addtional experiments


## Test
- **Step I.** Enter directory:
```
cd ./codes/example/TTRD3
```
- **Step II.** Modify the dataroots for train and val based on your actual root in
  - `./codes/example/TTRD3/options/TTRD3.yml`
- **Step III.** Start Testing:
```
sh val.sh
```

---

## Results
- AID test set:
<p align="center">
  <img src="figures/AID.png">

- RSD46-WHU、NWPU-RESISC45 and UCMerced_LandUse test set:
<p align="center">
  <img src="figures/Other_dataset.png">

- Comparison with SOTA Results
<p align="center">
  <img src="figures/Results.png">


## Acknowledgement
The code is based on [RDDM](https://github.com/nachifur/RDDM). We thank the authors for their excellent contributions.


## Contact
If you have any questions about our work, please contact [liyide23@mails.ucas.ac.cn](liyide23@mails.ucas.ac.cn).
