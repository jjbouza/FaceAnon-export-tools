import torch
from deep_privacy import deep_privacy_generator

def save_generator(net, inp, device):
    img, keypoints, z = inp["im"], inp["keypoints"], inp["z"]
    img, keypoints, z = img.to(device), keypoints.to(device), z.to(device)
    out = net(img, keypoints, z)
    model = torch.jit.trace(net, (img, keypoints, z))
    model.save("generator.pt")

device = torch.device("cpu")

# deep privacy generator
g = deep_privacy_generator.load_generator('./deep_privacy/default_cpu.ckpt', './deep_privacy/config_default.yml', device)
gen_inputs = torch.load("./deep_privacy/generator_inputs.pt")
save_generator(g, gen_inputs, device)
