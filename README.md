# Instance Segmentation With Mask-Supervised Polygonal Boundary Transformers

From [Justin Lazarow (UCSD, now at Apple)](),  [Weijian Xu (UCSD, now at Microsoft)](https://weijianxu.com), and [Zhuowen Tu (UCSD)](https://pages.ucsd.edu/~ztu).

This repository is an official implementation of the paper [Instance Segmentation With Mask-Supervised Polygonal Boundary Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Lazarow_Instance_Segmentation_With_Mask-Supervised_Polygonal_Boundary_Transformers_CVPR_2022_paper.pdf) presented at CVPR 2022.

## Introduction

BoundaryFormer aims to provide a simple baseline for _regression-based_ instance segmentation. Notably, we use Transformers to regress a fixed number of points along
a simple polygonal boundary. This process makes continuous predictions and is thus end-to-end differentiable. Our method differs from previous work in the field in two
main ways: our method can match Mask R-CNN in Mask AP for the first time and we impose no additional supervision or ground-truth requirements as Mask R-CNN. That is,
our method achieves parity in mask quality and supervision to mask-based baselines. We accomplish this by solely relying on a differentiable rasterization module (implemented in CUDA)
which only requires access to ground-truth masks. We hope this can serve to drive further work in this area.

<img src=".github/arch.png" width="300" >

## Installation

BoundaryFormer uses the same installation process as Detectron2. Please see [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html). This
should generally require something like:

``` shell
pip install -ve .
```

at the root of the source tree (as long as PyTorch, etc are installed correctly.

BoundaryFormer also uses the deformable attention modules introduced in [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR). If this
is already installed on your system, no action is needed. Otherwise, please build their modules:

```
git clone https://github.com/fundamentalvision/Deformable-DETR
cd Deformable-DETR/models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Getting Started

BoundaryFormer follows the general guidelines of Detectron2, however, it lives under ```projects/BoundaryFormer```.

Please make sure to set two additional environmental variables on your system:

``` shell

export DETECTRON2_DATASETS=/path/to/datasets
export DETECTRON2_OUTPUTS=/path/to/outputs
```

For instance, to train on COCO using an R50 backbone at a 1x schedule:

``` python
python projects/BoundaryFormer/train_net.py --num-gpus 8 --config-file projects/BoundaryFormer/configs/COCO-InstanceSegmentation/boundaryformer_rcnn_R_50_FPN_1x.yaml COMMENT "hello model"
```

If you do not have 8 GPUs, adjust --num-gpus and your BATCH_SIZE accordingly. BoundaryFormer is trained with AdamW and we find the square-root scaling law to work well (i.e., a batch size of 8 should only induce a sqrt(2) change in LR).

## Model Zoo

We release models for MS-COCO and Cityscapes.

## License

BoundaryFormer uses Detectron2 and is further released under the [Apache 2.0 license](LICENSE).

## Citing BoundaryFormer

If you use BoundaryFormer in your research, please use the following BibTeX entry.

```BibTeX
@InProceedings{Lazarow_2022_CVPR,
    author    = {Lazarow, Justin and Xu, Weijian and Tu, Zhuowen},
    title     = {Instance Segmentation With Mask-Supervised Polygonal Boundary Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4382-4391}
}
```
