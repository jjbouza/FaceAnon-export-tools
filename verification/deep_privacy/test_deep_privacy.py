import torch

import sys
sys.path.insert(1, 'test/')

from deep_privacy.config_parser import load_config
from deep_privacy.inference import infer, deep_privacy_anonymizer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from inout_generate import run_gt

def test(imgf, novel_img=True):
    if novel_img:
        run_gt(imgf)
    
    device = 'cpu'
    
    deep_privacyf = "../../deep_privacy/deep_privacy/deep_privacy.pt"
    model = torch.jit.load(deep_privacyf)
    pre_inputs = torch.load("preprocess_input.pt")
    img, keypoints, bbox= pre_inputs["im"], pre_inputs["keypoints"], pre_inputs["bbox"]
    img, keypoints, bbox = img[0].to(device), keypoints[0].to(device), bbox[0].to(device)

    out = model(img, keypoints, bbox, torch.tensor([keypoints.shape[0]]))
    
    plt.imshow(out)
    plt.show()
    
if __name__ == '__main__':
    test('images/sample.jpeg', True)
