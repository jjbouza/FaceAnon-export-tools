import torch

import sys
sys.path.insert(1, 'test/')

from deep_privacy.config_parser import load_config
from deep_privacy.inference import infer, deep_privacy_anonymizer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def run_gt(imgf):
    # run original DeepPrivacy modified to save results locally
    config = load_config("test/models/default/config.yml")
    checkpoint = torch.load("test/models/default/checkpoints/step_40000000.ckpt")
    generator = infer.init_generator(config, checkpoint)
    anonymizer = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator,
                                                               batch_size=1,
                                                               use_static_z=True,
                                                               keypoint_threshold=.1,
                                                               face_threshold=.6)

    anonymizer.anonymize_image_paths([imgf], ["example_anonymized.jpg"])


def test_preprocess(imgf):
    pass

def test_generator(imgf, novel_im=True):
    if novel_im:
        run_gt(imgf)
    
    # run torchscript DeepPrivacy
    generatorf = "../../deep_privacy/deep_privacy/generator.pt"
    generator_model = torch.jit.load(generatorf)
    inp = torch.load('generator_inputs.pt')
    im, keypoints, z = inp["im"], inp["keypoints"], inp["z"]
    out = []
    for i in range(im.shape[0]):
        out.append(generator_model(im[i:i+1], keypoints[i:i+1], z))
    out = torch.cat(out)    

    # compare
    out_gt = torch.load("generator_outputs.pt")
    if torch.abs(torch.mean( (out_gt - out))) < 0.005:
        print("Passed...")
    
if __name__ == '__main__':
    test_generator('sample3.jpeg', True)
