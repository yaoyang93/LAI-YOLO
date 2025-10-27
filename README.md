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

<img width="1213" height="354" alt="image" src="https://github.com/user-attachments/assets/464c58bb-bd84-4ef1-a9ad-f82a29ab1a64" />
