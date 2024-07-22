# U-Net
___

## Experiments

In order to compare performance of above implementations. There was utilized __Oxford III-T Pets__.
As a metric mean Intersection over Union (mIOU) was utilized. Models are compared on validation sets:

| __Loss Function__  | __U-Net V1__ | __U-Net V2__ |
|--------------------|--------------|--------------|
| Cross-entropy Loss | N/A          | N/A          |
| Dice Loss          | N/A          | N/A          |

Training procedure:  
- __Optimizer:__ SGD  
- __Learning rate:__ 1e-2
- __Momentum:__ 0.99
- __Weight decay:__ 1e-4
- __Batch size:__ 36


## References
___
[[1] Olaf Ronneberger, Philipp Fischer, Thomas Brox _U-Net: Convolutional Networks for Biomedical Image Segmentation_](https://arxiv.org/abs/1505.04597)  
[[2] Pytorch-UNet](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py)  
[[3] UNet for Building Segmentation (PyTorch)](https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch#Training-UNet)