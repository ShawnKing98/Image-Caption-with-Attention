# Image-Caption-with-Attention

Pytorch implementation of the paper *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*.

UMich EECS 545 final project.

Running guide:
- Download dataset from [COCO2014](https://cocodataset.org/#download) or [Vizwiz](https://vizwiz.org/tasks-and-datasets/image-captioning/)
- Download the python api of [COCO](https://github.com/cocodataset/cocoapi) or Vizwiz(https://github.com/Yinan-Zhao/vizwiz-caption)
- Run prepro.py to build the vocabulary and resize the pictures
- Run main.py to start training
- Run validate.py along with passed-in checkpoint path to do the test

