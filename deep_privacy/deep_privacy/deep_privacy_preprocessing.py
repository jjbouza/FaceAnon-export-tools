import torch
import torch.nn.functional as F

from math import floor

@torch.jit.script
def expand_bbox_simple(bbox, percentage=torch.tensor([0.4])):
    x0 : int = int(bbox[0].item())
    y0 : int = int(bbox[1].item())
    x1 : int = int(bbox[2].item())
    y1 : int = int(bbox[3].item())
    width = x1 - x0
    height = y1 - y0
    c2 = 2
    x_c = x0 + width/c2
    y_c = y0 + height/c2

    if width > height:
        avg_size = width
    else: 
        avg_size = height

    new_width = avg_size * (1 + percentage.item())

    x0 = x_c - floor(new_width/c2)
    y0 = y_c - floor(new_width/c2)
    x1 = x_c + floor(new_width/c2)
    y1 = y_c + floor(new_width/c2)

    return torch.tensor([x0, y0, x1, y1]).int()

@torch.jit.script
def pad_image(im, bbox):
    x0 : int = int(bbox[0].item())
    y0 : int = int(bbox[1].item())
    x1 : int = int(bbox[2].item())
    y1 : int = int(bbox[3].item())

    if x0 < 0:
        pad_im = torch.zeros(im.shape[0], abs(x0), im.shape[2]).int()
        im = torch.cat([pad_im, im], dim=1)
        x1 += abs(x0)
        x0 = 0
    if y0 < 0:
        pad_im = torch.zeros(abs(y0), im.shape[1], im.shape[2]).int()
        im = torch.cat([pad_im, im], dim=0)
        y1 += abs(y0)
        y0 = 0
    if x1 >= im.shape[1]:
        pad_im = torch.zeros(
            im.shape[0], x1 - im.shape[1] + 1, im.shape[2]).int()
        im = torch.cat([im, pad_im], dim=1)
    if y1 >= im.shape[0]:
        pad_im = torch.zeros(
            y1 - im.shape[0] + 1, im.shape[1], im.shape[2]).int()
        im = torch.cat([im, pad_im], dim=0)

    return im[y0:y1, x0:x1]


def cut_face(im, bbox, simple_expand=True):
    return pad_image(im, bbox)

@torch.jit.script
def shift_bbox(orig_bbox, expanded_bbox, new_imsize):
    x0 : int = int(orig_bbox[0].item())
    y0 : int = int(orig_bbox[1].item())
    x1 : int = int(orig_bbox[2].item())
    y1 : int = int(orig_bbox[3].item())

    x0e : int = int(expanded_bbox[0].item())
    y0e : int = int(expanded_bbox[1].item())
    x1e : int = int(expanded_bbox[2].item())
    y1e : int = int(expanded_bbox[3].item())
    
    new_imsize_ : int = int(new_imsize.item())

    x0, x1 = x0 - x0e, x1 - x0e
    y0, y1 = y0 - y0e, y1 - y0e
    w_ = x1e - x0e
    x0 = x0*new_imsize_/w_
    y0 = y0*new_imsize_/w_
    x1 = x1*new_imsize_/w_
    y1 = y1*new_imsize_/w_

    return torch.tensor([x0, y0, x1, y1])

def shift_and_scale_keypoint(keypoint, expanded_bbox):
    keypoint = keypoint.clone().float()
    keypoint[:, 0] -= expanded_bbox[0]
    keypoint[:, 1] -= expanded_bbox[1]
    w = expanded_bbox[2] - expanded_bbox[0]
    keypoint = keypoint / w
    return keypoint

@torch.jit.script
def cut_bounding_box(condition, bounding_boxes):
    bounding_boxes = bounding_boxes.clone()

    x0, y0, x1, y1 = [k.item() for k in bounding_boxes]
    if x0 >= x1 or y0 >= y1:
        return condition

    condition[y0:y1, x0:x1, :] = 128
    return condition

def pre_process(im, keypoint, bbox):
    expanded_bbox = expand_bbox_simple(bbox)
    to_replace = cut_face(im, expanded_bbox)

    imsize = torch.Tensor([128]).int()
    new_bbox = torch.floor(shift_bbox(bbox, expanded_bbox, imsize)).int()
    new_keypoint = shift_and_scale_keypoint(keypoint, expanded_bbox)

    to_replace = to_replace.permute(2,0,1).unsqueeze(0).float()
    to_replace = F.interpolate(to_replace, size=(imsize.item(),imsize.item()), mode='bilinear')
    to_replace = to_replace[0].permute(1,2,0).int()

    to_replace = cut_bounding_box(to_replace.clone(), new_bbox)
    torch_input = (to_replace.float()/255.0 * 2 - 1)
    torch_input = torch_input.permute(2,0,1).unsqueeze(0)

    return torch_input, new_keypoint.view(1,-1), expanded_bbox, new_bbox




