Project focused on gesture recognition using convolutional neural networks (LeNet-5 and ResNet-like backbones) with custom PyTorch data pipelines and image augmentations.

* Downloaded and explored the Zindi gesture dataset
* Removed duplicate samples and checked data quality
* Designed a train/validation split with 33% random validation data
* Implemented a custom PyTorch Dataset and DataLoader using OpenCV and pandas
* Rebuilt and trained the LeNet-5 architecture for gesture classification
* Implemented a training/validation loop with Cross Entropy loss and ROC AUC evaluation
* Trained a pretrained vision backbone (e.g. ResNet18) with a custom classification head
* Applied and evaluated augmentations with Albumentations
* Implemented and tested MixUp and CutMix augmentation strategies
