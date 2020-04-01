import torch
import torch.nn as nn
import torch.nn.functional as F

class preprocess_ssfd(nn.Module):
    def __init__(self):
        super(preprocess_ssfd, self).__init__()

    def forward(self, img):
        img = img[0].permute(1,2,0)
        img = img - torch.tensor([104, 117, 123])
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        return img

class postprocess_ssfd(nn.Module):
    def __init__(self):
        super(postprocess_ssfd, self).__init__()

    def decode(self, loc, priors, variances):
        # type: (Tensor, Tensor, List[float]) -> Tensor
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def forward(self, olist):
        # type: (Tuple[Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]) -> Tensor
        bboxlist: List[List[float]] = []

        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            poss = torch.where(ocls[:, 1, :, :] > 0.05)
            for i in range(poss[0].shape[0]):
                hindex = int(poss[0][i].item())
                windex = int(poss[1][i].item())

                axc = stride / 2 + windex * stride
                ayc = stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                value = [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
                priors = torch.tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)

                x1 = float(box[0][0].item())
                y1 = float(box[0][0].item())
                x2 = float(box[0][0].item())
                y2 = float(box[0][0].item())
                score = float(score.item())

                bboxlist.append([x1,y1,x2,y2,score])

        if 0 == len(bboxlist):
            bboxlist_ = torch.zeros(1,5)

        bboxlist_ = torch.tensor(bboxlist)

        return bboxlist_

def export_model_script(model, fname):
    print("Exporting ", fname, "...")
    model_instance = model()
    model_script = torch.jit.script(model_instance)
    model_script.save('processing_modules/'+fname+'.pt')
    print("Done exporting ", fname)

if __name__ == '__main__':
    # export models
    export_model_script(preprocess_ssfd, "preprocess_ssfd")
    export_model_script(postprocess_ssfd, "postprocess_ssfd")
    post_instance = postprocess_ssfd()




