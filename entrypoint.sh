#!/bin/bash

retinanet train retinanet_resnet34fpn.pth \
--backbone ResNet34FPN \
--iters 90000 \
--lr 0.0001 \
--batch 8 \
--with-dali \
--logdir tensorboard_logs  \
--images /data//coco/images/train2014/ \
--annotations /data//coco/annotations/instances_train2014.json \
--val-images /data/coco/images/val2014/ \
--val-annotations /data/coco/annotations/instances_val2014.json