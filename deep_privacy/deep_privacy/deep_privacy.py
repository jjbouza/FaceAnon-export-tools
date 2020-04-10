import deep_privacy.deep_privacy_preprocessing  as pre
import deep_privacy.deep_privacy_generator      as gen
import deep_privacy.deep_privacy_postprocessing as post

import torch
import torch.nn as nn

# combine preprocessing, generator and postprocessing into one module here.
class deep_privacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_z = torch.randn(1, 32, 4, 4)

        device = 'cpu'
        pre_inputs = torch.load("./deep_privacy/preprocess_input.pt")
        img, keypoints, bbox= pre_inputs["im"], pre_inputs["keypoints"], pre_inputs["bbox"]
        img, keypoints, bbox = img[0].to(device), keypoints[0].to(device), bbox[0].to(device)

        self.pre_process = torch.jit.trace(pre.pre_process, (img, keypoints[0], bbox[0]))

        gener = gen.load_generator('./deep_privacy/default_cpu.ckpt',
                './deep_privacy/config_default.yml', 'cpu')

        gen_inputs = torch.load("./deep_privacy/generator_inputs.pt")
        img, keypoints, z = gen_inputs["im"], gen_inputs["keypoints"], gen_inputs["z"]
        img, keypoints, z = img.to(device), keypoints.to(device), z.to(device)
        self.generator = torch.jit.trace(gener, (img, keypoints, z))
        
        post_inputs = torch.load("./deep_privacy/postprocess_inputs.pt")
        face_info, generated_faces, image = post_inputs["face_info"], post_inputs["generated_faces"], post_inputs["images"]
        img, out, expanded_bbox, bbox = image[0], generated_faces, face_info[0]["expanded_bbox"], face_info[0]["face_bbox"]
        img, expanded_bbox, bbox = torch.from_numpy(img), torch.from_numpy(expanded_bbox), torch.from_numpy(bbox)

        self.post_process = torch.jit.trace(post.post_process, (img, out, expanded_bbox, bbox))

    def new_z(self):
        self.static_z = torch.randn(1, 32, 4, 4)

    def forward(self, img, keypoints, bbox, n):
        # img: [H, W, 3]
        # keypoints: [n, 7, 2]
        # bbox: [n, 4]
        
        im = torch.tensor([0])

        for i in range(int(n[0].item())):
            torch_input, new_keypoint, expanded_bbox, new_bbox = self.pre_process(img, keypoints[i], bbox[i])
            out = self.generator(torch_input, new_keypoint, self.static_z)
            im = self.post_process(img, out, expanded_bbox, bbox[i])
        
        return im

