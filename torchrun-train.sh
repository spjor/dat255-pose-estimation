torchrun --nproc_per_node=cpu lib/train.py\
    --data-path data/coco --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 5 --device cpu\
    --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --workers 2