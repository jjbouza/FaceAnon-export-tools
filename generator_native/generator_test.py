import torch

import sys
sys.path.insert(1, 'test/')

from deep_privacy.config_parser import load_config
from deep_privacy.inference import infer, deep_privacy_anonymizer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def run_gt(imgf):
    # run original DeepPrivacy modified to save results locally
    config = load_config("test/models/isvc_large/config.yml")
    checkpoint = torch.load("test/models/isvc_large/checkpoints/step_40000000.ckpt")
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
    generatorf = "../extract_onnx/generator.pt"
    generator_model = torch.jit.load(generatorf)
    inp = torch.load('generator_inputs.pt')
    im, keypoints, z = inp["im"], inp["keypoints"], inp["z"]
    out = generator_model(im, keypoints, z)
    
    # compare
    out_gt = torch.load("generator_outputs.pt")
    if torch.abs(torch.mean( (out_gt - out))) < 0.05:
        print("Passed...")

    plt.imshow(out[0][1].detach().numpy())
    plt.show()
    
if __name__ == '__main__':
    test_generator('sample.jpeg', True)
