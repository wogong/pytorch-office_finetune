# PyTorch-Office_Finetune

A PyTorch implementation for fine-tuning AlexNet and ResNet on Office dataset.

## Environment

- Python 3.6
- PyTorch 0.4.0

## Result

|                |   A-W   |
| :------------: | :-----: |
| this(alexnet)  | 0.6000  |
| this(resnet50) | 0.7597  |

## Note

- alexnet pretrained model is converted  from caffe pretrained model (`bvlc_reference_caffenet.caffemodel`), using <https://github.com/leelabcnbc/pytorch-caffe-models>. Converted model can be download [here](https://www.dropbox.com/s/4x1hzbachpezx0w/alexnet_caffe.pth?dl=0), `inference.py` can be used as inference validation. `inference.py` under `caffe` directory is the Caffe version inference code.
- LRN layer is officially supported by PyTorch now
- Caffe's AlexNet implementation has different LRN/Pool layer order from original paper, this repo uses  conv -> pool -> LRN order (better results). Refer to <https://github.com/BVLC/caffe/issues/296> for details
- tried <https://github.com/jiecaoyu/pytorch_imagenet>, results is bad (<50%)
- tried torchvision pretrained alexnet model, results is bad (~54%))
- tried correct order of classifier layers, refer to <https://github.com/pytorch/vision/pull/550>, no improve
- fc/final0 and fc/final2 parameter init is important, best at 61%

## links

- <https://discuss.pytorch.org/t/pytorch-alexnet-not-as-good-as-original-alexnet-implementation/16680>
- <https://www.qin.ee/2018/06/25/da-office/>
