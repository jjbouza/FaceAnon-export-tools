import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn

class KeypointDetector(nn.Module):
    def __init__(self):
        super(KeypointDetector, self).__init__()
        self.keypointrcnn = keypointrcnn_resnet50_fpn(pretrained=True)

    def forward(self, x):
        x = self.keypointrcnn([x])
        kps = x[0]["keypoints"]
        scores = x[0]["scores"]

        return kps, scores
