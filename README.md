# 1.install

timm==1.0.7 

thop efficientnet_pytorch==0.7.1 

einops grad-cam==1.5.4 

dill==0.3.8 

albumentations==1.4.11

pytorch_wavelets==1.3.0 

PyWavelets==1.1.1 

opencv-python==4.12.0

# 2.Key Directory Structure

## 2.1 Configuration Files (ultralytics/cfg)

Model Architecture: Located in the LAI-YOLO subdirectory, containing detailed configuration files for network structures.

Dataset Settings: Stored in the datasets subdirectory, defining paths, classes, and related parameters.

Training/Testing Setup: Managed via default.yaml, including hyperparameters, optimizer settings, and other training configurations.

## 2.2 Module Implementations (ultralytics/nn)

# 3.Result graph

## 3.1 HeatMap
This is the heat map of YOLOv11-n
![Figure 11 (a)](https://github.com/user-attachments/assets/85f9a674-462a-473d-a733-61ab38b4b3ba)
This is the heat map of LAI-YOLO
![Figure 11 (b)](https://github.com/user-attachments/assets/316f4511-317e-4591-bc1f-3a32e4830075)

## 3.2 Visual Analysis
This is the heat map of YOLOv11-n
![Figure 10 b_1](https://github.com/user-attachments/assets/d3330ef0-7dde-4de9-88b7-f431c212508a)
This is the heat map of LAI-YOLO
![Figure 10 C_1](https://github.com/user-attachments/assets/56fcc325-1d80-45c0-b6d5-d644cfb6e1a1)

# 4.Download link of the dataset used in the article
VisDrone : https://github.com/VisDrone/VisDrone-Dataset
UAVDT : https://sites.google.com/view/grli-uavdt/
AI-TOD : https://github.com/jwwangchn/AI-TOD
CODrone : https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV
