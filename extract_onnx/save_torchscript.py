import torch
from keypointrcnn import KeypointDetector

# save the keypoint rcnn torchscript

def save_torchscript_keypointrcnn(net):
    keypointrcnn_ts = torch.jit.script(net)
    keypointrcnn_ts.save("keypointrcnn.pt")

keypointrcnn = KeypointDetector()
save_torchscript_keypointrcnn(keypointrcnn)
