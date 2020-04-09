import torch

device = torch.device("cpu")

def map_location(ckptf, outf):
    ckpt_mapped = torch.load(ckptf, map_location=device)
    torch.save(ckpt_mapped, outf)

if __name__ == "__main__":
    map_location("default.ckpt", "default_cpu.ckpt")
    map_location("large.ckpt", "large_cpu.ckpt")

