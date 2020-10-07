# Block Shuffle: A Method for High-resolution Fast Style Transfer with Limited Memory

## Introduction

For high-resolution images, most mobile devices and personal computers cannot stylize them due to memory limitations. To solve this problem, we proposed a novel method named block shuffle, which can stylize high-resolution images with limited memory. In our experiments, we used [Logan Engstrom's implementation of Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer) as the baseline. In this repository, we provided the source code and 16 trained models. In addition, we developed an Android demo app, if you are interested in it, please click [here](https://github.com/czczup/MusesArt).

### Requirements

1. CUDA 9, cudnn 7
2. Python 3.6
3. Python packages: tensorflow-gpu==1.9, opencv-python, numpy

### Citation
```
@article{ma2020block,
  title={Block Shuffle: A Method for High-resolution Fast Style Transfer with Limited Memory},
  author={Ma, Weifeng and Chen, Zhe and Ji, Caoting},
  journal={IEEE Access},
  year={2020},
  publisher={IEEE}
}
```

## High-resolution Image Stylization

### Trained Fast Style Transfer Models

|        [#01](models/01)         |        [#02](models/02)         |        [#03](models/03)         |        [#04](models/04)         |        [#05](models/05)         |        [#06](models/06)         |        [#07](models/07)         |        [#08](models/08)         |
| :-----------------------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: |
| ![style](examples/style/01.jpg) | ![style](examples/style/02.jpg) | ![style](examples/style/03.jpg) | ![style](examples/style/04.jpg) | ![style](examples/style/05.jpg) | ![style](examples/style/06.jpg) | ![style](examples/style/07.jpg) | ![style](examples/style/08.jpg) |
|        [#09](models/09)         |        [#10](models/10)         |        [#11](models/11)         |        [#12](models/12)         |        [#13](models/13)         |        [#14](models/14)         |        [#15](models/15)         |        [#16](models/16)         |
| ![style](examples/style/09.jpg) | ![style](examples/style/10.jpg) | ![style](examples/style/11.jpg) | ![style](examples/style/12.jpg) | ![style](examples/style/13.jpg) | ![style](examples/style/14.jpg) | ![style](examples/style/15.jpg) | ![style](examples/style/16.jpg) |

### Baseline

Use `baseline.py` to stylize a high-resolution image. A GPU with 12GB memory can stylize up to 4000\*4000 images (if your GPU doesn't have enough memory, it will throw an OOM error). Example usage:

```sh
python baseline.py --input examples/content/xxx.jpg \
  --output examples/result/xxx.jpg \
  --model models/01/model.pb \
  --gpu 0
```

### Feathering-based Method

Use `feathering-based.py` to stylize a high-resolution image. This method is very simple, but it doesn't work well. Example usage:

```sh
python feathering-based.py --input examples/content/xxx.jpg \
  --output examples/result/xxx.jpg \
  --model models/01/model.pb \
  --gpu 0
```

### Block Shuffle Method (Ours)

Use `block_shuffle.py` to stylize a high-resolution image. In our experiments, we set the max-width to 1000. If your GPU cannot stylize a 1000\*1000 image, you can change this parameter to a smaller value. Example usage:

```sh
python block_shuffle.py --input examples/content/xxx.jpg \
  --output examples/result/xxx.jpg \
  --model models/01/model.pb \
  --max-width 1000 \
  --gpu 0
```



## Training Style Transfer Networks

### Download the COCO2014 Dataset and the Pre-trained VGG-19

We provided 16 trained fast style transfer models. If you want to train a new model, please download the COCO2014 dataset and the pre-trained VGG-19. If not, you can skip this step.

You can run `setup.sh` to download the COCO2014 dataset and the pre-trained VGG-19. Or you can download them from the following link (place them in `data/`):

1. [COCO2014 dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)
2. [pre-trained VGG-19](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)

### Train

Use `style.py` to train a new style transfer network. Training takes 26 hours on a Nvidia Telas K80 GPU. Example usage:

```sh
python style.py --style examples/style/style.jpg \
  --checkpoint-dir checkpoint/style01 \
  --test examples/content/chicago.jpg \
  --test-dir checkpoint/style01
```

### Export

Use `export.py` to export .pb files. Example usage:

```sh
python export.py --input checkpoint/xxx/fns.ckpt \
  --output checkpoint/xxx/models.pb \
  --gpu 0 
```



## Some Experimental Results

|             Style              |                  Content                   |                  Baseline                   |                  Baseline+Feathering                   |              Baseline+Block Shuffle (Ours)               |
| :----------------------------: | :----------------------------------------: | :-----------------------------------------: | :----------------------------------------------------: | :------------------------------------------------------: |
| ![style](results/1/style.jpg)  | ![style](results/1/thumbnail/content.jpg)  | ![style](results/1/thumbnail/baseline.jpg)  | ![style](results/1/thumbnail/baseline+feathering.jpg)  | ![style](results/1/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/2/style.jpg)  | ![style](results/2/thumbnail/content.jpg)  | ![style](results/2/thumbnail/baseline.jpg)  | ![style](results/2/thumbnail/baseline+feathering.jpg)  | ![style](results/2/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/3/style.jpg)  | ![style](results/3/thumbnail/content.jpg)  | ![style](results/3/thumbnail/baseline.jpg)  | ![style](results/3/thumbnail/baseline+feathering.jpg)  | ![style](results/3/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/4/style.jpg)  | ![style](results/4/thumbnail/content.jpg)  | ![style](results/4/thumbnail/baseline.jpg)  | ![style](results/4/thumbnail/baseline+feathering.jpg)  | ![style](results/4/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/5/style.jpg)  | ![style](results/5/thumbnail/content.jpg)  | ![style](results/5/thumbnail/baseline.jpg)  | ![style](results/5/thumbnail/baseline+feathering.jpg)  | ![style](results/5/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/6/style.jpg)  | ![style](results/6/thumbnail/content.jpg)  | ![style](results/6/thumbnail/baseline.jpg)  | ![style](results/6/thumbnail/baseline+feathering.jpg)  | ![style](results/6/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/7/style.jpg)  | ![style](results/7/thumbnail/content.jpg)  | ![style](results/7/thumbnail/baseline.jpg)  | ![style](results/7/thumbnail/baseline+feathering.jpg)  | ![style](results/7/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/8/style.jpg)  | ![style](results/8/thumbnail/content.jpg)  | ![style](results/8/thumbnail/baseline.jpg)  | ![style](results/8/thumbnail/baseline+feathering.jpg)  | ![style](results/8/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/9/style.jpg)  | ![style](results/9/thumbnail/content.jpg)  | ![style](results/9/thumbnail/baseline.jpg)  | ![style](results/9/thumbnail/baseline+feathering.jpg)  | ![style](results/9/thumbnail/baseline+blockshuffle.jpg)  |
| ![style](results/10/style.jpg) | ![style](results/10/thumbnail/content.jpg) | ![style](results/10/thumbnail/baseline.jpg) | ![style](results/10/thumbnail/baseline+feathering.jpg) | ![style](results/10/thumbnail/baseline+blockshuffle.jpg) |
| ![style](results/11/style.jpg) | ![style](results/11/thumbnail/content.jpg) | ![style](results/11/thumbnail/baseline.jpg) | ![style](results/11/thumbnail/baseline+feathering.jpg) | ![style](results/11/thumbnail/baseline+blockshuffle.jpg) |
| ![style](results/12/style.jpg) | ![style](results/12/thumbnail/content.jpg) | ![style](results/12/thumbnail/baseline.jpg) | ![style](results/12/thumbnail/baseline+feathering.jpg) | ![style](results/12/thumbnail/baseline+blockshuffle.jpg) |
| ![style](results/13/style.jpg) | ![style](results/13/thumbnail/content.jpg) | ![style](results/13/thumbnail/baseline.jpg) | ![style](results/13/thumbnail/baseline+feathering.jpg) | ![style](results/13/thumbnail/baseline+blockshuffle.jpg) |
| ![style](results/14/style.jpg) | ![style](results/14/thumbnail/content.jpg) | ![style](results/14/thumbnail/baseline.jpg) | ![style](results/14/thumbnail/baseline+feathering.jpg) | ![style](results/14/thumbnail/baseline+blockshuffle.jpg) |



