<p align="center">

  <h1 align="center">LEARN FROM ZOOM: DECOUPLED SUPERVISED CONTRASTIVE LEARNING FOR WCE IMAGE CLASSIFICATION</h1>
  <p align="center">
  <div align="center">
    <img src="images/structure.png", width="600">
  </div>
  <div align="center">
  <br>
    <a href="https://arxiv.org/abs/2401.05771">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
  </div>
  </p>
</p>

## Introduction
Accurate lesion classification in Wireless Capsule Endoscopy (WCE) images is crucial for early detection of gastrointestinal (GI) cancers. However, challenges like small lesions and background interference make this task difficult. To overcome these hurdles, we propose Decoupled Supervised Contrastive Learning for WCE image classification. Using zoomed-in WCE images generated by Saliency Augmentor, our approach achieves a remarkable 92.01% overall accuracy within 10 epochs, surpassing the prior state-of-the-art by 0.72% on two publicly accessible WCE datasets.

## Requirements
The usual installation steps involve the following commands, they should set up the correct CUDA version and all the python packages:
```bash
conda create -n DSCL python=3.10
conda activate DSCL
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Data and Structure
We evaluated our method on a combined dataset of 3022 images, merging CAD-CAP (1812 images) and KID (1210 images) datasets. If you want to use the same data, you need to submit a request to the data owner.
```bash
--data
  --Fold-0
    --train
    --val
  --Fold-1
    --train
    --val
```

## Train
Here are example commands for training:
```bash
cd scripts
python txt.py
cd ..
sh run_supcon.sh
sh run_suplinear.sh
```

## Acknowledgements
This code is developed based on [Saliency-Sampler](https://github.com/recasens/Saliency-Sampler/tree/master) and art of the code is borrowed from [SupContrast](https://github.com/HobbitLong/SupContrast)

## Citation
If you find our work useful in your research or if you use parts of this code, please consider citing our paper:
```bash
@misc{qiu2024learn,
      title={Learn From Zoom: Decoupled Supervised Contrastive Learning For WCE Image Classification}, 
      author={Kunpeng Qiu and Zhiying Zhou and Yongxin Guo},
      year={2024},
      eprint={2401.05771},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
