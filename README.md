# Object detection in challenging lighting conditions

This repository contains source code for diploma thesis titled: Object detection in challenging lighting conditions


![sample img](readme_imgs/img1.png)

## objective
Framework for:
- object detection with [ExDark dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- evaluation different models and fine-tuning them

## Use cases
- comparing not-fine-tuned object detectors
- comparing fine-tuned object detectors in standard manner
- evaluation of image-enhancement techniques
- evaluation of model trained for challenging conditions with techniques like: siamese network, DANN (domain adversarial neural networks) or auxiliary reconstruction tasl   

## key files descriptions
```shell
/data # directory with raw immutable data and script to convert labels and bounding boxes
config.py # file with parameters like batch size
datasets.py # torch dataset adjusted object detection task
eval.py # metrics calculation like mAP
train.py # training loop
env.yml # conda environment configuration file
```