import ctypes

import torch
from skimage import io
import coremltools

from test.detection.sfd.sfd_detector import SFDDetector

def correct_coreml_dims(olist):
    out = []
    for ol in olist:
        if ol.shape[0] == ol.shape[1]:
            out.append(ol[0])
        else:
            out.append(ol)
    return out

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
    fssfdcml = '../extract_onnx/sfd_detector.mlmodel'

    preprocess = torch.jit.load(fpreprocess)
    postprocess = torch.jit.load(fpostprocess)
    #ssfd = onnxruntime.InferenceSession(fssfd)
    ssfd_corml = coremltools.models.MLModel(fssfdcml)

    img_o = preprocess(torch.from_numpy(img).float())
    #input_name = ssfd.get_inputs()[0].name
    #res = ssfd.run(None, {input_name: img_o.numpy()})
    res = ssfd_corml.predict({'input_img': img_o.numpy()})
    names = ['ol1', 'ol2', 'ol3', 'ol4', 'ol5', 'ol6', 'ol7', 'ol8', 'ol9', 'ol10', 'ol11', 'ol12']
    olist = [torch.from_numpy(res[n]) for n in names]
    olist = correct_coreml_dims(olist)
    out= postprocess(*olist)

    return out, img_o, olist


img = io.imread("sample.jpeg")
if img.shape[2] == 4:
    img = img[:,:,:3]
img = img.reshape((1,)+img.shape).transpose(0,3,1,2)[:,:,:,:][:, :, :, :].copy()
out1, img1, olist1= run_align_torchscript(img)
out2, img2, olist2= run_align_pytorch(img)

if (img1 == img2).all():
    print("Preprocessing passed...")
else: 
    print("DID NOT PASS PREPROCESSING...")

olist_passed = [torch.allclose(ol1, ol2, atol=1*10**(-1)) for ol1, ol2 in zip(olist1, olist2)]

for i in range(len(olist_passed)):
    if olist_passed[i]:
        print("olist[", i, "] passed...")
    else:
        print("olist[", i, "] DID NOT PASS...")

if torch.allclose(out1, out2, atol=1*10**(-1)):
    print("Postprocessing passed... ALL GOOD!")
else:
    print("DID NOT PASS POSTPROCESSING...")

print(out1)
print(out2)
