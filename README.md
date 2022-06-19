# Instance Segmentation With Mask-Supervised Polygonal Boundary Transformers (Justin Lazarow, Weijian Xu, and Zhuowen Tu), CVPR 2022

This is the code release for our CVPR 2022 paper:

_Instance Segmentation With Mask-Supervised Polygonal Boundary Transformers (Justin Lazarow, Weijian Xu, and Zhuowen Tu)_

which we refer to as **BoundaryFormer**.

BoundaryFormer aims to provide a simple baseline for _regression-based_ instance segmentation. Notably, we use Transformers to regress a fixed number of points along
a simple polygonal boundary. This process makes continuous predictions and is thus end-to-end differentiable. Our method differs from previous work in the field in two
main ways: our method can match Mask R-CNN in Mask AP for the first time and we impose no additional supervision or ground-truth requirements as Mask R-CNN. That is,
our method achieves parity in mask quality and supervision to mask-based baselines. We accomplish this by solely relying on a differentiable rasterization module (implemented in CUDA)
which only requires access to ground-truth masks. We hope this can serve to drive further work in this area.

Paper: [CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Lazarow_Instance_Segmentation_With_Mask-Supervised_Polygonal_Boundary_Transformers_CVPR_2022_paper.pdf)

## Installation

BoundaryFormer uses the same installation process as Detectron2. Please see [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

TODO

## Model Zoo and Baselines

TODO

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
