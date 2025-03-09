# Learning from Failure: Training Debiased Classifier from Biased Classifier
This repository provides a PyTorch implementation of "Learning from Failure: Training Debiased Classifier from Biased Classifier" by Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin (NeurIPS 2020).

# Training vanilla resnet
`python -m Models.Vanilla.rn --mode online `
# Training LfF
After editing the hyerparameters in the config dict in `main`

`python -m train `
# Testing the models
>CelebA

`python -m test --weights_path` Models\Weights\... 

>ResNet

`python -m test -s --weights_path Models\Weights\CelebA-ResNet\resnet18_BlondHair_best.pth`
# 📌 Reference
Paper: [arXiv Link](https://arxiv.org/abs/2007.02561)
