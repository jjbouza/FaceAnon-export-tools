import torch
import torch.nn.functional as F

def stitch_face(im, expanded_bbox, generated_face, bbox_to_extract, original_bbox):
    # Ugly but works....
    # if tight_stitch is set to true, only the part of the image inside original_bbox is updated
    x0e, y0e, x1e, y1e = expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]
    x0o, y0o, x1o, y1o = bbox_to_extract[0], bbox_to_extract[1], bbox_to_extract[2], bbox_to_extract[3]
    x0, y0, x1, y1 = original_bbox[0], original_bbox[1], original_bbox[2], original_bbox[3]

    image_mask = torch.ones_like(im).to(torch.bool)
    mask_single_face = image_mask[y0e:y1e, x0e:x1e]

    to_replace = im[y0e:y1e, x0e:x1e].int()
    generated_face = generated_face[y0o:y1o, x0o:x1o]

    to_replace[mask_single_face] = generated_face[mask_single_face]
    im[y0e:y1e, x0e:x1e] = to_replace
    image_mask[y0:y1, x0:x1, :] = 0

    return im

@torch.jit.script
def replace_face(im, generated_face, original_bbox, expanded_bbox):

    bbox_to_extract = torch.tensor([0, 0, generated_face.shape[1], generated_face.shape[0]])

    for i in range(2):
        if expanded_bbox[i] < 0:
            bbox_to_extract[i] -= expanded_bbox[i]
            expanded_bbox[i] = 0

    if expanded_bbox[2] > im.shape[1]:
        diff = expanded_bbox[2] - im.shape[1]
        bbox_to_extract[2] -= diff
        expanded_bbox[2] = im.shape[1]

    if expanded_bbox[3] > im.shape[0]:
        diff = expanded_bbox[3] - im.shape[0]
        bbox_to_extract[3] -= diff
        expanded_bbox[3] = im.shape[0]

    im = stitch_face(im, expanded_bbox, generated_face, bbox_to_extract, original_bbox)
    return im

def post_process(im, generated_face, expanded_bbox, original_bbox):

    # denorm image
    generated_face = torch.clamp((generated_face+1)/2, 0, 1)
    generated_face = (generated_face*255)

    orig_imsize = expanded_bbox[2] - expanded_bbox[0]

    generated_face = F.interpolate(generated_face, size=(orig_imsize, orig_imsize), mode='bilinear')
    generated_face = generated_face.int()
    
    generated_face = generated_face[0].permute(1,2,0)

    im = replace_face(im, generated_face, original_bbox, expanded_bbox)
    
    # output image
    return im



