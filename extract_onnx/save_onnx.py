import numpy

import torch
from torch.utils.model_zoo import load_url
from skimage import io

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}

def save_onnx(net, img, device):
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)
    img = torch.from_numpy(img).float().to(device)
    olist = net(img)
    olist = torch.onnx.export(net, img, f='sfd_detector.onnx', export_params=True)

#import s3df net

from net_s3fd import s3fd

device = torch.device("cuda")

model_weights = load_url(models_urls['s3fd'])

face_detector = s3fd()
face_detector.load_state_dict(model_weights)
face_detector.to(device)
face_detector.eval()

img = io.imread('./aflw-test.jpg')

save_onnx(face_detector, img, device)