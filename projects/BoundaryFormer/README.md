# Instances masks are what you need: Segmentation parity from object boundaries

Justin Lazarow, Weijian Xu, Zhuowen Tu

## Training

To train a model with 8 GPUs run:
```bash
cd /path/to/detectron2/projects/BoundaryFormer
python train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/boundaryformer_rcnn_R_50_FPN_1x.yaml
```
