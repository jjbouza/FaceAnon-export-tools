import torch

import sys
sys.path.insert(1, 'test/')

from deep_privacy.config_parser import load_config
from deep_privacy.inference import infer, deep_privacy_anonymizer

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

if __name__ == '__main__':
    run_gt('sample3.jpeg')
