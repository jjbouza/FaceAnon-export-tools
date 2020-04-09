import torch
import torch.nn as nn
import torch.nn.functional as F

class preprocess_ssfd(nn.Module):
    def __init__(self):
        super(preprocess_ssfd, self).__init__()

    def forward(self, img):
        # [1, 3, H, W] with channels in bgr order.
        img = img[0].permute(1,2,0)
        img = img - torch.tensor([104, 117, 123])
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        return img.contiguous()

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
    
    def nms(self, dets):
        thresh = 0.3

        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.flip(scores.argsort(), dims=[0])
        keep  = []
        idx = torch.ones(order.shape[0], dtype=torch.bool)

        while idx.any():
            i = order[idx][0]
            keep.append(i)
            xx1, yy1 = torch.max(x1[i], x1[order[idx][1:]]), torch.max(y1[i], y1[order[idx][1:]])
            xx2, yy2 = torch.min(x2[i], x2[order[idx][1:]]), torch.min(y2[i], y2[order[idx][1:]])
            w, h = torch.clamp(xx2 - xx1 + 1, 0.0), torch.clamp( yy2 - yy1 + 1, 0.0)
            ovr = w * h / (areas[i] + areas[order[idx][1:]] - w * h)
            
            order = order[idx]
            inds = torch.where(ovr <= thresh)[0]+1
            mask = torch.zeros_like(order)
            mask = mask.scatter_(0, inds, 1.).to(dtype=torch.bool)
            idx = mask
            

        return torch.stack(keep).float()

    def forward(self, ol1, ol2, ol3, ol4, ol5, ol6, ol7, ol8, ol9, ol10, ol11, ol12):
        # type: (Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor) -> Tensor
        olist = [ol1,ol2,ol3,ol4,ol5,ol6,ol7,ol8,ol9,ol10,ol11,ol12]
        bboxlist: List[List[float]] = []
        poss = []

        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            poss = torch.where(ocls[:, 1, :, :] > 0.05)
            for i in range(poss[0].shape[0]):
                hindex = int(poss[1][i].item())
                windex = int(poss[2][i].item())
                
                axc = stride / 2 + windex * stride
                ayc = stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                value = [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
                priors = torch.tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)

                x1 = float(box[0][0].item())
                y1 = float(box[0][1].item())
                x2 = float(box[0][2].item())
                y2 = float(box[0][3].item())
                score = float(score.item())

                bboxlist.append([x1,y1,x2,y2,score])

        if 0 == len(bboxlist):
            bboxlist_ = torch.zeros(1,5)

        bboxlist_ = torch.tensor(bboxlist).float()
        keep  = self.nms(bboxlist_).long()
        bboxlist_ = bboxlist_[keep, :]
        bboxlist_ = bboxlist_[torch.where(bboxlist_[:,4] > 0.5)[0]]

        return bboxlist_.clone()

def export_model_script(model, fname):
    print("Exporting ", fname, "...")
    model_instance = model()
    model_script = torch.jit.script(model_instance)
    model_script.save('./outputs/'+fname+'.pt')
    print("Done exporting ", fname)

if __name__ == '__main__':
    # export models
    export_model_script(preprocess_ssfd, "preprocess_ssfd")
    export_model_script(postprocess_ssfd, "postprocess_ssfd")
    post_instance = postprocess_ssfd()




