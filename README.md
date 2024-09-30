# Learning Exposure Correction in Dynamic Scenes 

### (ACM MM 2024 Oral)

This repository is the official implementation of the VECNet, where more implementation details are presented.

<hr />

### 1. Dataset Download
Please download our DIME dataset on [BaiduCloud](https://pan.baidu.com/s/1uU2P_fhDGVDjhCggEuasWw?pwd=298u).

### 2. Requirements
```
# clone this repo
git clone https://github.com/kravrolens/VECNet.git
cd VECNet

# create an environment with python >= 3.9
conda env create -f env.yaml
```

### 3. Testing
```
bash scripts/test.sh
```
### 4. Training
```
bash scripts/train.sh
```

<hr />

### Citation
If you find this work useful for your research, please consider citing:
``` 
@article{DBLP:journals/corr/abs-2402-17296,
  author       = {Jin Liu and
                  Bo Wang and
                  Chuanming Wang and
                  Huiyuan Fu and
                  Huadong Ma},
  title        = {Learning Exposure Correction in Dynamic Scenes},
  journal      = {CoRR},
  volume       = {abs/2402.17296},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.17296},
  doi          = {10.48550/ARXIV.2402.17296},
  eprinttype    = {arXiv},
  eprint       = {2402.17296},
  timestamp    = {Mon, 25 Mar 2024 15:38:17 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2402-17296.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
