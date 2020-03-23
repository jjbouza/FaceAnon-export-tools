#include "processing.h"

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

torch::Tensor postprocess(std::vector<torch::Tensor> olist){
    std::vector<float> bboxlist;

    //apply softmax to even elements of olist...
    for(int i = 0; i < olist.size()/2; i++){
        olist[i*2] = torch::nn::functional::softmax(olist[i*2], 1);
    }

    for(int i = 0; i < olist.size()/2; i++){
        torch::Tensor ocls = olist[i*2];
        torch::Tensor oreg = olist[i*2+1];

        int stride = pow(2, i+2);
        int anchor = stride*4;
        //to do...
    }

    return olist[0];
}