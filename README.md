# Image-Caption-with-Attention

Pytorch implementation of the paper *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*.

UMich Fall 2021 EECS 545 final project.

Running guide:
- Download dataset from [COCO2014](https://cocodataset.org/#download) or [Vizwiz](https://vizwiz.org/tasks-and-datasets/image-captioning/)
- Download the python api of [COCO](https://github.com/cocodataset/cocoapi) or [Vizwiz](https://github.com/Yinan-Zhao/vizwiz-caption)
- Run prepro.py to build the vocabulary and resize the pictures
- Run main.py to start training
- Run validate.py along with passed-in checkpoint & vocabulary path to do the test. This script can predict a caption given a image, as well as give bleu scores on given validation set.

Our network parameters and vocabulary file for both datasets have been uploaded to [Google drive](https://drive.google.com/drive/folders/1G77T-LKDiabX4T6qNecKY1IFEjZfzCDn?usp=sharing), trained for 3 days on COCO (~30 epoches) and Vizwiz (~100 epoches) using Greatlakes servers. 
