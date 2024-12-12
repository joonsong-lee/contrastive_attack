# contrastive_attack
This repo is for attacking black-box face swapping models by attacking face recognition models.
The reason why attack face recognition model is face recognition model is used as a feature extractor in face swapping model.

## Preparation
If you wanna use my code and don't wanna modify anything. download all of belows and move all of files and directories to designated directory or you can just rename it to designated dir name.
### Face Recognition models
I did experiment and wrote code based on 2 different weights from [ArcFace](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/README.md) , [AdaFace](https://github.com/mk-minchul/AdaFace), [Facenet](https://github.com/timesler/facenet-pytorch).

For arcface, you must download above insight face repo and just copy all of under reocognition-arcface_torch to ./arcface_torch. also, download [2pretrained weights(ms1mv3r50,glint360Kr50)](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBc3dwc0RPMnRvTktxMGxXWTY5dk41OEdSNm13P2U9cDlPdjVk&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC) and rename glint backbone to "gli_backbone.pth". both backbones are needed to move into ./arcface_torch/

For AdaFace, download it by above link. copy all of under main repo to ./AdaFace/ or you know.

download WebFace4M r50 pretrained model to ./AdaFace and rename it to "adaface_ir50_webface4m.ckpt"

For facenet, you can simply use python library by pip install facenet-pytorch.
### Facial attribute transfer model
I did experiment and wrote code based on [STD-GAN](https://github.com/XuyangGuo/STD-GAN).

download and copy all of under the main repo to ./STD_GAN and follow the instruction of preparation including pretrained model on above repo.

you can find detail setting of this model in [augmen.py](https://github.com/joonsong-lee/contrastive_attack/blob/main/augmen.py).
### Face detection model
For face allignment and face cropping, i used [retinaface-pytorch(3rd party implement)](https://github.com/supernotman/RetinaFace_Pytorch).

download and copy all of under the main repo to ./RetinaFace_Pytorch and download pretrained weight under the same link and rename it to "model.pt".
### Requirements
Check Requirements.txt

## Inference
Adaface model is needed to get bgr image, there is additional line for Adaface. so you must set the key of Adaface is ada.
### Dataset
if you wanna do single inference, there is no need to download extra datasets. but, if you wanna do some batch job. you need to modify code of run.py.

i did experiment with huggingface datasets library. if you wanna do with it, you just remove comment sections relating with huggingface datasets.
### Single inference
```

python single_inference.py --i_path /data/source.py --save_path /data/result
```
if you wanna change options, please check single_inference.py

### Batch job
Use run.py and find options for experiment.


