#include "processing.h"

#include <vector>
#include <math.h>

torch::Tensor preprocess(torch::Tensor img){

    // rgb constant shift
    torch::Tensor C = torch::ones({3});
    C[0] = 104; C[1] = 117; C[2] = 123;
    img = img - C;

    // channels to front, add dim 1 to front
    img = img.permute({2, 0, 1});
    img = img.unsqueeze(0);

    return img;
}


torch::Tensor decode(torch::Tensor loc, torch::Tensor priors, std::vector<float> variances){
    torch::Tensor e1 = priors.narrow(1, 0, 2) + loc.narrow(1, 0, 2) * priors.narrow(1, 0, 2) * variances[0];
    torch::Tensor e2 = priors.narrow(1, 2, 2) * torch::exp(loc.narrow(1, 2, 2)*variances[1]);

    torch::Tensor boxes = torch::cat({e1, e2}, 1);
    boxes.narrow(1, 0, 2) -= boxes.narrow(1, 2, 2) / 2;
    boxes.narrow(1, 2, 2) += boxes.narrow(1, 0, 2);
    
    return boxes.view({4});
}

torch::Tensor postprocess(std::vector<torch::Tensor> olist){
    std::vector<std::vector<float>> bboxlist;

    //apply softmax to even elements of olist...
    for(int i = 0; i < olist.size()/2; i++){
        olist[i*2] = torch::nn::functional::softmax(olist[i*2], 1);
    }

    for(int i = 0; i < olist.size()/2; i++){
        torch::Tensor ocls = olist[i*2];
        torch::Tensor oreg = olist[i*2+1];

        int stride = pow(2, i+2);
        int anchor = stride*4;
        

        std::vector<torch::Tensor> poss = torch::where(torch::narrow(ocls, 1, 0, 1).squeeze(1) > 0.05);

        for(int i = 0; i < poss[0].sizes()[0]; i++){
            int Iindex = poss[0][i].item<int>();
            int hindex = poss[1][i].item<int>();
            int windex = poss[2][i].item<int>();

            float axc = stride / 2 + windex*stride;
            float ayc = stride / 2 + hindex*stride;

            float score = ocls[0][1][hindex][windex].item<float>();

            torch::Tensor loc = oreg.narrow(0, 0, 1).narrow(2, hindex, 1).narrow(3, windex, 1).contiguous().view({1,4});

            torch::Tensor priors = torch::ones({1, 4});
            priors[0][0] = axc; priors[0][1] = ayc; priors[0][2] = stride * 4; priors[0][3] = stride * 4;

            std::vector<float> variances = {0.1, 0.2};
            torch::Tensor box = decode(loc, priors, variances);
            bboxlist.push_back({box[0].item<float>(), box[1].item<float>(), box[2].item<float>(), box[3].item<float>(), score});
        }
    }

    if(bboxlist.size() == 0)
        return torch::zeros({1,5});

    // create torch tensor
    torch::Tensor result = torch::zeros({(long int)bboxlist.size(), (long int)bboxlist[0].size()});
    
    for(int i = 0; i < bboxlist.size(); i++)
        for(int j = 0; j < bboxlist[0].size(); j++)
            result[i][j] = bboxlist[i][j];

    return result;
}




