import torch

from test.deep_privacy.config_parser import load_config
from test.deep_privacy.inference import infer, deep_privacy_anonymizer

def test_generator():
    # run original DeepPrivacy modified to save results locally
    config = load_config("test/models/isvc_large/config.yml")
    checkpoint = torch.load("test/models/isvc_large/checkpoints/step_40000000.ckpt")
    generator = infer.init_generator(config, checkpoint)

    generatorf = "../extract_onnx/generator.pt"
    generator_model = torch.jit.load(generatorf)

    

