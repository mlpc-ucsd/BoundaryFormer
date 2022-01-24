# Current Experiments

## COCO ResNet-50 @ 1x

|Notes|Weight decay|AP<sub>mask</sub>|AP<sub>50</sub>|AP<sub>75</sub>|AP<sub>s</sub>|AP<sub>m</sub>|AP<sub>l</sub>|AP<sub>bbox</sub>|
|---|---|---|---|---|---|---|---|---|
|Mask R-CNN|0.05|35.8370|56.8447|38.4427|17.0811|38.2020|51.7407|38.7646|
|BoundaryFormer (0.01, No detach)|0.05|35.9809|56.3854|38.3393|17.6114|38.3080|52.6425|38.5038|


## COCO ResNet-101 @ 3x

* The main takeaway is that weight decay must be scaled with larger
  backbones and/or longer schedules. 0.05 is not sufficient.
* It is also possible that weight decay should be larger for
  BoundaryFormer than Mask R-CNN (TBD).

|Notes|Weight decay|DETR aug|AP<sub>mask</sub>|AP<sub>50</sub>|AP<sub>75</sub>|AP<sub>s</sub>|AP<sub>m</sub>|AP<sub>l</sub>|AP<sub>bbox</sub>|
|---|---|---|---|---|---|---|---|---|---|
|Mask R-CNN|0.10|✖|38.5895|60.0198|41.5645|19.2262|40.9211|55.7962|42.3495|
|Mask R-CNN|0.15|✖|38.7663|60.2468|41.6288|19.1839|41.3015|55.6481|42.5476|
|Mask R-CNN|0.20|✖|39.2627|60.8258|42.3553|19.5025|41.7188|55.9546|43.1227|
|BoundaryFormer|0.05|✖|37.5579|58.6683|40.0921|18.8434|39.4459|54.3999|40.8340|
|BoundaryFormer|0.10|✔|39.3945|61.0269|42.5773|19.5278|41.8965|56.9687|43.1426|
