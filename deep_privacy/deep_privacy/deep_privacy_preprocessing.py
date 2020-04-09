import torch

def expand_bbox_simple(bbox, percentage=0.4):
    bbox = bbox.float()
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    width = x1 - x0
    height = y1 - y0
    x_c = int(x0) + width//2.0
    y_c = int(y0) + height//2.0
    avg_size = max(width, height)
    new_width = avg_size * (1 + percentage)
    x0 = x_c - new_width//2
    y0 = y_c - new_width//2
    x1 = x_c + new_width//2
    y1 = y_c + new_width//2
    return torch.tensor([x0, y0, x1, y1]).int()

@torch.jit.script
def pad_image(im, bbox):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

    if x0 < 0:
        pad_im = torch.zeros((im.shape[0], abs(x0), im.shape[2])).int()
        im = torch.cat((pad_im, im), axis=1)
        x1 += abs(x0)
        x0 = 0
    if y0 < 0:
        pad_im = torch.zeros((abs(y0), im.shape[1], im.shape[2])).int()
        im = torch.cat((pad_im, im), axis=0)
        y1 += abs(y0)
        y0 = 0
    if x1 >= im.shape[1]:
        pad_im = torch.zeros(
            (im.shape[0], x1 - im.shape[1] + 1, im.shape[2])).int()
        im = torch.cat((im, pad_im), axis=1)
    if y1 >= im.shape[0]:
        pad_im = torch.zeros(
            (y1 - im.shape[0] + 1, im.shape[1], im.shape[2])).int()
        im = torch.cat((im, pad_im), axis=0)
    return im[y0:y1, x0:x1]


def cut_face(im, bbox, simple_expand=True):
    return pad_image(im, bbox)

def shift_bbox(orig_bbox, expanded_bbox, new_imsize):
    orig_bbox = orig_bbox.float()
    expanded_bbox = expanded_bbox.float()

    x0, y0, x1, y1 = orig_bbox[0], orig_bbox[1], orig_bbox[2], orig_bbox[3]
    x0e, y0e, x1e, y1e = expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]

    x0, x1 = x0 - x0e, x1 - x0e
    y0, y1 = y0 - y0e, y1 - y0e
    w_ = x1e - x0e
    x0, y0, x1, y1 = [int(k*new_imsize/w_) for k in [x0, y0, x1, y1]]
    return torch.tensor([x0, y0, x1, y1])

def shift_and_scale_keypoint(keypoint, expanded_bbox):
    keypoint = keypoint.clone().float()
    keypoint[:, 0] -= expanded_bbox[0]
    keypoint[:, 1] -= expanded_bbox[1]
    w = expanded_bbox[2] - expanded_bbox[0]
    keypoint = keypoint / w
    return keypoint

def cut_bounding_box(condition, bounding_boxes):
    bounding_boxes = bounding_boxes.clone()

    x0, y0, x1, y1 = [k.item() for k in bounding_boxes]
    if x0 >= x1 or y0 >= y1:
        return condition

    condition[y0:y1, x0:x1, :] = 128
    return condition

def pre_process(im, keypoint, bbox, imsize):
    expanded_bbox = expand_bbox_simple(bbox)
    to_replace = cut_face(im, expanded_bbox)
    new_bbox = shift_bbox(bbox, expanded_bbox, imsize)

    new_keypoint = shift_and_scale_keypoint(keypoint, expanded_bbox)

    # to_replace = cv2.resize(to_replace, (imsize, imsize))

    to_replace = cut_bounding_box(to_replace.copy(), new_bbox)
    torch_input = to_replace * 2 - 1

    return torch_input, new_keypoint, expanded_bbox, new_bbox




