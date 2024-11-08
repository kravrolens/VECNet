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
@inproceedings{liu2024learning,
  title={Learning Exposure Correction in Dynamic Scenes},
  author={Liu, Jin and Wang, Bo and Wang, Chuanming and Fu, Huiyuan and Ma, Huadong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={3858--3866},
  year={2024}
}
```
