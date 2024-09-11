# SegNet
___

_SegNet_ is an example of CNN with encoder-decoder architecture. 
The encoder part is identical to 13 convolutional layers of the VGG16 network. 
The only difference is addition of _Batch Normalization_ layers between each _Conv_ layers and _ReLU_ layers.
The goal of decoder part is to decode encoded representation of input to final response - 
classified input image pixels. 
Its architecture is reversed compared to the encoder. 
The only difference is that _Pooling_ layers are replaced by _Upsampling_ layers and 
moved at the beginning of the blocks. Additionally, indices from _Pooling_ layers are 
forwarded to corresponding _Upsampling_ layers.

Below picture depicts the original architecture of SegNet:

![Architcture](pictures/architecture.png)

## Experiments

In order to compare performance of above implementations. There was utilized __Oxford III-T Pets__.
As a metric mean Intersection over Union (mIOU) was utilized. Models are compared on validation sets:

| __Loss Function__  | __SegNet__ |
|--------------------|------------|
| Cross-entropy Loss | 0.8736     |
| Dice Loss          | 0.8904     |
| BCE Loss           | 0.5380     |
| Dice BCE Loss      | 0.8782     |
| Focal Loss         | 0.0292     |
| Tversky Loss       | 0.8453     |
| Focal Tversky Loss | 0.8312     |

## References
___
[[1] Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla _SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation_](https://arxiv.org/abs/1511.00561)  
[[2] Pytorch SegNet & DeepLabV3 Training](https://www.kaggle.com/code/robinreni/pytorch-segnet-deeplabv3-training)
