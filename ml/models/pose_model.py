import torchvision
from torch import nn

from ml.models.convnet_backbone import ConvNetBackbone
from ml.models.deconv_head import DeconvHead
from ml.models.resnet_backbone import get_resnet18_backbone, get_resnet34_backbone, get_resnet50_backbone, \
    get_resnet101_backbone


class PoseEstimationModel(nn.Module):

    def __init__(self, backbone, head, name='pose_estimation_model'):
        super(PoseEstimationModel, self).__init__()
        self.name = name
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)

        return out


def get_pose_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 pose estimation model with deconvolution head."""
    backbone = get_resnet18_backbone(pretrained=pretrained, **kwargs)
    head = DeconvHead(512, 17)

    name = "pose_resnet18"
    if pretrained:
        name += "_pretrained"


    return PoseEstimationModel(backbone, head, name=name)

def get_pose_resnet34(pretrained=False, **kwargs):
    backbone = get_resnet34_backbone(pretrained=pretrained, **kwargs)
    head = DeconvHead(512, 17)

    name = "pose_resnet34"
    if pretrained:
        name += "_pretrained"

    return PoseEstimationModel(backbone, head, name=name)

def get_pose_resnet50(pretrained=False, **kwargs):
    backbone = get_resnet50_backbone(pretrained=pretrained, **kwargs)
    head = DeconvHead(2048, 17)

    name = "pose_resnet50"
    if pretrained:
        name += "_pretrained"

    return PoseEstimationModel(backbone, head, name=name)

def get_pose_resnet101(pretrained=False, **kwargs):
    backbone = get_resnet101_backbone(pretrained=pretrained, **kwargs)
    head = DeconvHead(2048, 17)

    name = "pose_resnet101"
    if pretrained:
        name += "_pretrained"

    return PoseEstimationModel(backbone, head, name=name)

def get_pose_resnet152(pretrained=False, **kwargs):
    backbone = get_resnet50_backbone(pretrained=pretrained, **kwargs)
    head = DeconvHead(2048, 17)

    name = "pose_resnet152"
    if pretrained:
        name += "_pretrained"

    return PoseEstimationModel(backbone, head, name=name)

def get_pose_convnet():
    backbone = ConvNetBackbone()
    head = DeconvHead(1024, 17, 2)

    return PoseEstimationModel(backbone, head, name='pose_convnet')

if __name__ == '__main__':
    model = get_pose_resnet50(pretrained=True)
    print(model)

    model = get_pose_resnet18(pretrained=False)
    print(model)

    model = get_pose_resnet34(pretrained=False)
    print(model)

    model = get_pose_convnet()
    print(model)

