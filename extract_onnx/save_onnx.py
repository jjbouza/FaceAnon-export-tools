import numpy
import torch
from torch.utils.model_zoo import load_url
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from skimage import io

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}

def save_onnx_ssfd(net, img, device):
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)
    img = torch.from_numpy(img).float().to(device)
    out = net(img)
    olist = torch.onnx.export(net, 
                              img,
                              f='sfd_detector.onnx', 
                              export_params=True, 
                              input_names=['input_img'],
                              output_names=['ol1', 'ol2', 'ol3', 'ol4', 'ol5', 'ol6', 
                                            'ol7', 'ol8', 'ol9', 'ol10', 'ol11', 'ol12'],
                              example_outputs=out,
                              dynamic_axes={'input_img': {2:'height',3:'width'}})

def save_onnx_keypointrcnn(net, img, device):
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().to(device).unsqueeze(0)
    keypoints = net(img)
    keypoints = torch.onnx.export(net, 
                                  img, 
                                  f='keypointrcnn.onnx', 
                                  export_params=True,
                                  input_names=['input_img'],
                                  output_names=['keypoints', 'scores'],
                                  opset_version=11)

from net_s3fd import s3fd

device = torch.device("cpu")
img = io.imread('./aflw-test.jpg')

# ssfd model
model_weights = load_url(models_urls['s3fd'])
face_detector = s3fd()
face_detector.load_state_dict(model_weights)
face_detector.to(device)
face_detector.eval()
save_onnx_ssfd(face_detector, img, device)

