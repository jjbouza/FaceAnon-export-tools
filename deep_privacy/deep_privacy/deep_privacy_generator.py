import torch
import torch.nn as nn

import numpy as np

import yaml
from collections import namedtuple


class ProgressiveBaseModel(nn.Module):

    def __init__(self, pose_size, start_channel_size, image_channels):
        super().__init__()

        self.transition_channels = [
            start_channel_size,
            start_channel_size,
            start_channel_size,
            start_channel_size//2,
            start_channel_size//4,
            start_channel_size//8,
            start_channel_size//16,
            start_channel_size//32,
        ]
        self.transition_channels = [x // 8 * 8
                                    for x in self.transition_channels]
        self.image_channels = image_channels
        self.transition_value = 1.0
        self.num_poses = pose_size // 2
        self.current_imsize = 4
        self.transition_step = 0
        self.prev_channel_extension = start_channel_size

    def extend(self):
        self.transition_value = 0.0
        self.prev_channel_extension = self.transition_channels[self.transition_step]
        self.transition_step += 1
        self.current_imsize *= 2


    def load_state_dict(self, ckpt):
        if "transition_step" in ckpt.keys():
            for i in range(ckpt["transition_step"]):
                self.extend()
            self.transition_value = ckpt["transition_value"]

        super().load_state_dict(ckpt["parameters"])

def get_transition_value(x_old, x_new, transition_variable):
    return torch.lerp(x_old, x_new, transition_variable)


def generate_pose_channel_images_baseline(min_imsize, max_imsize, device, pose_information, dtype):
    batch_size = pose_information.shape[0]
    if pose_information.shape[1] == 2:
        pose_images = []
        imsize = min_imsize
        while imsize <= max_imsize:
            new_im = torch.zeros(
                (batch_size, 1, imsize, imsize), dtype=dtype, device=device)
            imsize *= 2
            pose_images.append(new_im)
        return pose_images
    num_poses = pose_information.shape[1] // 2
    pose_x = pose_information[:, range(0, pose_information.shape[1], 2)].view(-1)
    pose_y = pose_information[:, range(1, pose_information.shape[1], 2)].view(-1)
    if (max_imsize, batch_size) not in batch_indexes.keys():
        batch_indexes[(max_imsize, batch_size)] = torch.cat(
            [torch.ones(num_poses, dtype=torch.long)*k for k in range(batch_size)])
        pose_indexes[(max_imsize, batch_size)] = torch.arange(
            0, num_poses).repeat(batch_size)
    batch_idx = batch_indexes[(max_imsize, batch_size)]
    pose_idx = pose_indexes[(max_imsize, batch_size)].clone()
    # All poses that are outside image, we move to the last pose channel
    illegal_mask = ((pose_x < 0) + (pose_x >= 1.0) +
                    (pose_y < 0) + (pose_y >= 1.0)) != 0
    pose_idx[illegal_mask] = num_poses
    pose_x[illegal_mask] = 0
    pose_y[illegal_mask] = 0
    pose_images = []
    imsize = min_imsize
    while imsize <= max_imsize:
        new_im = torch.zeros((batch_size, num_poses+1, imsize, imsize),
                             dtype=dtype, device=device)

        px = (pose_x * imsize).long()
        py = (pose_y * imsize).long()
        new_im[batch_idx, pose_idx, py, px] = 1
        new_im = new_im[:, :-1]  # Remove "throwaway" channel
        pose_images.append(new_im)
        imsize *= 2
    return pose_images

@torch.jit.script
def generate_pose_channel_images(min_imsize, max_imsize, pose_information):
    # type: (int, int, Tensor) -> List[Tensor]
    batch_size = pose_information.shape[0]
    #if pose_information.shape[1] == 2:
    #    pose_images = []
    #    imsize = min_imsize
    #    while imsize <= max_imsize:
    #        new_im = torch.zeros(
    #            (batch_size, 1, imsize, imsize)).float()
    #        imsize *= 2
    #        pose_images.append(new_im)
    #    return pose_images
    num_poses = pose_information.shape[1] // 2

    even_indices = torch.arange(0, pose_information.shape[1], 2)
    odd_indices = even_indices+1
    pose_x = pose_information[:, even_indices].view(-1)
    pose_y = pose_information[:,  odd_indices].view(-1)

    batch_idx = torch.cat([torch.ones(num_poses, dtype=torch.long)*k for k in range(batch_size)])
    pose_idx =  torch.arange(0, num_poses).repeat(batch_size)

    # All poses that are outside image, we move to the last pose channel
    illegal_mask = ((pose_x < 0) + (pose_x >= 1.0) +
                    (pose_y < 0) + (pose_y >= 1.0)) != 0

    pose_idx[illegal_mask] = torch.tensor([num_poses])
    pose_x[illegal_mask] = torch.tensor([0]).float()
    pose_y[illegal_mask] = torch.tensor([0]).float()
    pose_images = []
    imsize = min_imsize
    while imsize <= max_imsize:
        new_im = torch.zeros((batch_size, num_poses+1, imsize, imsize)).float()

        px = (pose_x * imsize).long()
        py = (pose_y * imsize).long()
        new_im[batch_idx, pose_idx, py, px] = torch.tensor([1]).float()
        new_im = new_im[:, :-1]  # Remove "throwaway" channel
        pose_images.append(new_im)
        imsize *= 2
    return pose_images

# https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then 
    it will multiply the scale for every forward pass
    """

    def __init__(self, inCh, outCh, kernelSize, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh,
                              kernel_size=kernelSize, stride=1, padding=padding)

        # new bias to use after wscale
        bias = self.conv.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))
        self.conv.bias = None

        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:])  # Leave out # of op filters
        self.wtScale = gain/np.sqrt(fanIn)

        # init
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        self.name = f"{self.conv.__class__.__name__}-{str(convShape)}"

    def forward(self, x):
        return self.conv(x * self.wtScale) + self.bias

    def __repr__(self):
        return self.__class__.__name__ + self.name

class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        factor = ((x**2).mean(dim=1, keepdim=True) + 1e-8)**0.5
        return x / factor


class UpSamplingBlock(nn.Module):

    def __init__(self):
        super(UpSamplingBlock, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        WSConv2d(in_dim, out_dim, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2),
        PixelwiseNormalization()
    )


class UnetDownSamplingBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(in_dim, out_dim, 3, 1),
            conv_bn_relu(out_dim, out_dim, 3, 1),
        )

    def forward(self, x):
        return self.model(x)


class UnetUpsamplingBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(in_dim, out_dim, 3, 1),
            conv_bn_relu(out_dim, out_dim, 3, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Generator(ProgressiveBaseModel):

    def __init__(self,
                 pose_size,
                 start_channel_dim,
                 image_channels):
        super().__init__(pose_size, start_channel_dim, image_channels)
        # Transition blockss
        self.to_rgb_new = WSConv2d(
            start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = WSConv2d(
            start_channel_dim, self.image_channels, 1, 0)

        self.core_blocks_down = nn.ModuleList([
            UnetDownSamplingBlock(start_channel_dim, start_channel_dim)
        ])
        self.core_blocks_up = nn.ModuleList([
            nn.Sequential(
                conv_bn_relu(start_channel_dim+self.num_poses +
                             32, start_channel_dim, 1, 0),
                UnetUpsamplingBlock(start_channel_dim, start_channel_dim)
            )
        ])

        self.new_up = nn.Sequential()
        self.old_up = nn.Sequential()
        self.new_down = nn.Sequential()
        self.from_rgb_new = conv_bn_relu(
            self.image_channels, start_channel_dim, 1, 0)
        self.from_rgb_old = conv_bn_relu(
            self.image_channels, start_channel_dim, 1, 0)

        self.upsampling = UpSamplingBlock()
        self.downsampling = nn.AvgPool2d(2)

    def extend(self):
        output_dim = self.transition_channels[self.transition_step]
        print("extending G", output_dim)
        # Downsampling module

        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
            self.from_rgb_new
        )
        if self.transition_step == 0:
            core_block_up = nn.Sequential(
                self.core_blocks_up[0],
                UpSamplingBlock()
            )
            self.core_blocks_up = nn.ModuleList([core_block_up])
        else:
            core_blocks_down = nn.ModuleList()

            core_blocks_down.append(self.new_down)
            first = [nn.AvgPool2d(2)] + \
                list(self.core_blocks_down[0].children())
            core_blocks_down.append(nn.Sequential(*first))
            core_blocks_down.extend(self.core_blocks_down[1:])

            self.core_blocks_down = core_blocks_down
            new_up_blocks = list(self.new_up.children()) + [UpSamplingBlock()]
            self.new_up = nn.Sequential(*new_up_blocks)
            self.core_blocks_up.append(self.new_up)

        self.from_rgb_new = conv_bn_relu(self.image_channels, output_dim, 1, 0)
        self.new_down = nn.Sequential(
            UnetDownSamplingBlock(output_dim, self.prev_channel_extension)
        )
        self.new_down = self.new_down
        # Upsampling modules
        self.to_rgb_old = self.to_rgb_new
        self.to_rgb_new = WSConv2d(output_dim, self.image_channels, 1, 0)

        self.new_up = nn.Sequential(
            conv_bn_relu(self.prev_channel_extension*2 +
                         self.num_poses, self.prev_channel_extension, 1, 0),
            UnetUpsamplingBlock(self.prev_channel_extension, output_dim)
        )
        super().extend()

    def new_parameters(self):
        new_paramters = list(self.new_down.parameters()) + \
            list(self.to_rgb_new.parameters())
        new_paramters += list(self.new_up.parameters()) + \
            list(self.from_rgb_new.parameters())
        return new_paramters

    def generate_latent_variable(self, *args):
        if len(args) == 1:
            x_in = args[0]
            return torch.randn(x_in.shape[0], 32, 4, 4,
                               device=x_in.device,
                               dtype=x_in.dtype)
        elif len(args) == 3:
            batch_size, device, dtype = args
            return torch.randn(batch_size, 32, 4, 4,
                               device=device,
                               dtype=dtype)
        raise ValueError(
            f"Expected either x_in or (batch_size, device, dtype. Got: {args}")

    def forward(self, x_in, pose_info, z=None):
        if z is None:
            z = self.generate_latent_variable(x_in)
        unet_skips = []

        new_down = self.from_rgb_new(x_in)
        new_down = self.new_down(new_down)
        unet_skips.append(new_down)
        new_down = self.downsampling(new_down)
        x = new_down

        for block in self.core_blocks_down[:-1]:
            x = block(x)
            unet_skips.append(x)
        x = self.core_blocks_down[-1](x)
        pose_channels = generate_pose_channel_images(4,
                                                     self.current_imsize,
                                                     pose_info)
        x = torch.cat((x, pose_channels[0], z), dim=1)
        x = self.core_blocks_up[0](x)
        for idx, block in enumerate(self.core_blocks_up[1:]):
            skip_x = unet_skips[-idx-1]
            x = torch.cat((x, skip_x, pose_channels[idx+1]), dim=1)
            x = block(x)


        x = torch.cat((x, unet_skips[0], pose_channels[-1]), dim=1)
        x_new = self.new_up(x)
        x_new = self.to_rgb_new(x_new)
        x = x_new

        return x

def convert_config(name, config):
    for key, value in config.items():
        if isinstance(value, dict) and key != "batch_size_schedule":

            config[key] = convert_config(key, value)
    return namedtuple(name, config.keys())(*config.values())

def load_config(config_path):
    with open(config_path, "r") as cfg_file:
        config = yaml.safe_load(cfg_file)
    return convert_config("Config", config)


def load_generator(ckptf, configf, device):
    ckpt = torch.load(ckptf, device)
    imsize = ckpt["current_imsize"]
    config = load_config(configf)

    g = Generator(
        config.models.pose_size,
        config.models.start_channel_size,
        config.models.image_channels
    )
    
    #g.load_state_dict(ckpt["G"])
    g.load_state_dict(ckpt["running_average_generator"])
    g.to(device)
    g.eval()

    return g

if __name__ == "__main__":
    dev = torch.device('cpu')
    g = load_generator("./default_cpu.ckpt", "config_default.yml", dev)
    




