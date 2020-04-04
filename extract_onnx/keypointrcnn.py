import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn

class KeypointDetector(nn.Module):
    def __init__(self):
        super(KeypointDetector, self).__init__()
        self.keypointrcnn = keypointrcnn_resnet50_fpn(pretrained=True)
        self.keypointrcnn.eval()
        self.keypoint_threshold = 0.3

    def forward(self, x):
        y = self.keypointrcnn([x])[1][0]
        kps = y["keypoints"]
        scores = y["scores"]

        mask = scores > self.keypoint_threshold
        kps_  = kps[mask, :, :2]

        return kps_.contiguous(), scores.contiguous()
