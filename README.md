# Human Activity Recognition with State Space Models

## Introduction
This repository provides the official PyTorch implementation of the method described in the paper: "TRIS-HAR: Transmissive Reconfigurable Intelligent Surfaces-assisted Cognitive Wireless Human Activity Recognition Using State Space Models".

Available at arXiv.

## Requirements
1. Install `PyTorch` and `torchvision` (we use `pytorch==2.2.2` and `torchvision==0.17.2`).

## Datasets
The public datasets (UT-HAR & NTU-Fi) can be available from Google Drive: [Public datasets](https://drive.google.com/drive/folders/14XOGHL0kUGrLw7APDR1QOLNS0Merboql?usp=drive_link).

Our dataset (HUST-HAR) can be available from China Mobile Drive: [HUST-HAR](https://caiyun.139.com/m/i?165CkGy8W6n4u), password: ahyc.

HUST-HAR includes six activities: lie down, pick up, sit down, stand, stand up, and walk. Each action fold contains six sub-fold (subject 1-6). Each subject repeats 100 times.

For example, standup1.mat is a complex matrix, and the dimension is $270 \times 5000$.

Dimension 270: $3 \times 3 \times 30$, three transmitter antennas, three receiver antennas, 30 subcarriers per antenna pair.

Dimension 5000: a sampling rate of 1000 Hz over a 5-second interval.

**UT-HAR**

[A Survey on Behavior Recognition Using WiFi Channel State Information](https://ieeexplore.ieee.org/document/8067693)

[Wifi_Activity_Recognition @ github](https://github.com/ermongroup/Wifi_Activity_Recognition)

**NTU-Fi**

[EfficientFi: Toward Large-Scale Lightweight WiFi Sensing via CSI Compression](https://ieeexplore.ieee.org/document/9667414)

[EfficientFi @ github](https://github.com/NTU-AIoT-Lab/EfficientFi)

Thanks to [Xinyan Chen](https://github.com/xyanchen). We use the functions `UT_HAR_dataset` and `CSI_Dataset` to process UT-HAR and NTU-Fi datasets for training. You can find the functions at [WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark).

## Run
Coming soon


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
