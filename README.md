## DSCL

Code for [**LEARN FROM ZOOM: DECOUPLED SUPERVISED CONTRASTIVE LEARNING FOR WCE IMAGE CLASSIFICATION**](https://arxiv.org/abs/2401.05771)

### Environment

```bash
conda create -n DSCL python=3.10
conda activate DSCL
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Data Structure
```bash
--data
  --Fold-0
    -train
    -val
  -Fold-1
    -train
    -val
```

### Train
```bash
cd scripts
python txt.py
cd ..
sh run_supcon.sh
sh run_suplinear.sh
```

### Citation
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

### Acknowledgements
Some codes from [Saliency-Sampler](https://github.com/recasens/Saliency-Sampler/tree/master) and [SupContrast](https://github.com/HobbitLong/SupContrast)
