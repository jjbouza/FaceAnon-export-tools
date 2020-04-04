import ctypes

import torch
import onnxruntime
from skimage import io

from test.detection.sfd.sfd_detector import SFDDetector

def run_align_pytorch(img):
    dev = "cpu"
    sfd = SFDDetector(dev)
    img = torch.from_numpy(img).float()[0].permute(1,2,0)
    bboxlist, img, olist= sfd.detect_from_image(img)
    
    return torch.tensor(bboxlist), img, olist

def run_align_torchscript(img):
    fpreprocess = '../processing_torchscript/processing_modules/preprocess_ssfd.pt'
    fpostprocess ='../processing_torchscript/processing_modules/postprocess_ssfd.pt' 
    fssfd = '../extract_onnx/sfd_detector.onnx'

    preprocess = torch.jit.load(fpreprocess)
    postprocess = torch.jit.load(fpostprocess)
    ssfd = onnxruntime.InferenceSession(fssfd)

    img_o = preprocess(torch.from_numpy(img).float())
    input_name = ssfd.get_inputs()[0].name
    res = ssfd.run(None, {input_name: img_o.numpy()})
    olist = [torch.from_numpy(ol) for ol in res]
    out= postprocess(*olist)

    return out, img_o, olist


img = io.imread("img.jpg").reshape(1,450,450,3).transpose(0,3,1,2)[:,:,:,:][:, ::-1, :, :].copy()
out1, img1, olist1= run_align_torchscript(img)
out2, img2, olist2= run_align_pytorch(img)

if (img1 == img2).all():
    print("Preprocessing passed...")
else: 
    print("DID NOT PASS PREPROCESSING...")

olist_passed = [torch.allclose(ol1, ol2, atol=1*10**(-4)) for ol1, ol2 in zip(olist1, olist2)]

for i in range(len(olist_passed)):
    if olist_passed[i]:
        print("olist[", i, "] passed...")
    else:
        print("olist[", i, "] DID NOT PASS...")

if torch.allclose(out1, out2, atol=1*10**(-4)):
    print("Postprocessing passed... ALL GOOD!")
else:
    print("DID NOT PASS POSTPROCESSING...")
