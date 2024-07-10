# Human Activity Recognition with State Space Models

## Introduction
This repository provides the official PyTorch implementation of the method described in the paper: "TRIS-HAR: Transmissive Reconfigurable Intelligent Surfaces-assisted Cognitive Wireless Human Activity Recognition Using State Space Models".

## Requirements
1. Install `PyTorch` and `torchvision` (we use `pytorch==2.2.2` and `torchvision==0.17.2`).

## Datasets
The datasets (UT-HAR & NTU-Fi) can be available from Google Drive: [Public datasets](https://drive.google.com/drive/folders/14XOGHL0kUGrLw7APDR1QOLNS0Merboql?usp=drive_link).

**UT-HAR**
[A Survey on Behavior Recognition Using WiFi Channel State Information](https://ieeexplore.ieee.org/document/8067693)
[Wifi_Activity_Recognition @ github](https://github.com/ermongroup/Wifi_Activity_Recognition)

**NTU-Fi**
[EfficientFi: Toward Large-Scale Lightweight WiFi Sensing via CSI Compression](https://ieeexplore.ieee.org/document/9667414)
[EfficientFi @ github](https://github.com/NTU-AIoT-Lab/EfficientFi)

Thanks to [Xinyan Chen](https://github.com/xyanchen). We use the functions `UT_HAR_dataset` and `CSI_Dataset` to process UT-HAR and NTU-Fi datasets for training. You can find the functions at [WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark).

## Reference
```
@article{yousefi2017survey,
  title={A survey on behavior recognition using WiFi channel state information},
  author={Yousefi, Siamak and Narui, Hirokazu and Dayal, Sankalp and Ermon, Stefano and Valaee, Shahrokh},
  journal={IEEE Communications Magazine},
  volume={55},
  number={10},
  pages={98--104},
  year={2017},
  publisher={IEEE}
}

@article{yang2022efficientfi,
  title={EfficientFi: Toward large-scale lightweight WiFi sensing via CSI compression},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Wang, Dazhuo and Xu, Qianwen and Xie, Lihua},
  journal={IEEE Internet of Things Journal},
  volume={9},
  number={15},
  pages={13086--13095},
  year={2022},
  publisher={IEEE}
}

@article{yang2023sensefi,
  title={SenseFi: A library and benchmark on deep-learning-empowered WiFi human sensing},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Lu, Chris Xiaoxuan and Wang, Dazhuo and Sun, Sumei and Xie, Lihua},
  journal={Patterns},
  volume={4},
  number={3},
  year={2023},
  publisher={Elsevier}
}
```