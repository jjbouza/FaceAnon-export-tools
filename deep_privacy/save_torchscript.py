import torch
from deep_privacy import deep_privacy_generator
from deep_privacy import deep_privacy_preprocessing
from deep_privacy import deep_privacy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_deep_privacy(net, inp, device):
    img, keypoints, bbox= inp["im"], inp["keypoints"], inp["bbox"]
    img, keypoints, bbox = img[0].to(device), keypoints[0].to(device), bbox[0].to(device)

    model = torch.jit.script(net)
    model.save("./deep_privacy/deep_privacy.pt")
    print("Saved deep_privacy")

def save_preprocessing(block, inp, device):
    img, keypoints, bbox= inp["im"], inp["keypoints"], inp["bbox"]
    img, keypoints, bbox = img[0].to(device), keypoints[0].to(device), bbox[0].to(device)
    model = torch.jit.trace(block, (img, keypoints[0], bbox[0]))
    model.save("./deep_privacy/preprocessing.pt")


def save_generator(net, inp, device):
    img, keypoints, z = inp["im"], inp["keypoints"], inp["z"]
    img, keypoints, z = img.to(device), keypoints.to(device), z.to(device)
    model = torch.jit.trace(net, (img[0:1], keypoints[0:1], z))
    model.save("./deep_privacy/generator.pt")

def save_postprocessing(net, inp, device):
    img, keypoints, z = None


device = torch.device("cpu")

def deep_privacy_():
    net = deep_privacy.deep_privacy()
    inputs = torch.load("./deep_privacy/preprocess_input.pt")
    save_deep_privacy(net, inputs, device)

def preprocess_():
    pre = deep_privacy_preprocessing.pre_process
    inputs = torch.load("./deep_privacy/preprocess_input.pt")
    save_preprocessing(pre, inputs, device)

def generator_():
    g = deep_privacy_generator.load_generator('./deep_privacy/default_cpu.ckpt', './deep_privacy/config_default.yml', device)
    gen_inputs = torch.load("./deep_privacy/generator_inputs.pt")
    save_generator(g, gen_inputs, device)

if __name__=='__main__':
    deep_privacy_()



