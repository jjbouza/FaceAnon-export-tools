#include "processing_wrapper.h"

#include <torch/torch.h>

float* preprocess_wrapper(float* img){
    torch::Tensor img_tensor = torch::from_blob(img, {450,450,3}, torch::kFloat);
    torch::Tensor img_preprocessed = preprocess(img_tensor);
    return img_preprocessed.data_ptr<float>();
}

float* postprocess_wrapper(float* olist0,
                           float* olist1,
                           float* olist2,
                           float* olist3,
                           float* olist4,
                           float* olist5,
                           float* olist6,
                           float* olist7,
                           float* olist8,
                           float* olist9,
                           float* olist10,
                           float* olist11,
                           int* num_boxes){

    torch::Tensor olist0_t = torch::from_blob(olist0, {1, 2, 112, 112}, torch::kFloat);
    torch::Tensor olist1_t = torch::from_blob(olist1, {1, 4, 112, 112}, torch::kFloat);
    torch::Tensor olist2_t = torch::from_blob(olist2, {1, 2, 56, 56}, torch::kFloat);
    torch::Tensor olist3_t = torch::from_blob(olist3, {1, 4, 56, 56}, torch::kFloat);
    torch::Tensor olist4_t = torch::from_blob(olist4, {1, 2, 56, 56}, torch::kFloat);
    torch::Tensor olist5_t = torch::from_blob(olist5, {1, 4, 56, 56}, torch::kFloat);
    torch::Tensor olist6_t = torch::from_blob(olist6, {1, 2, 56, 56}, torch::kFloat);
    torch::Tensor olist7_t = torch::from_blob(olist7, {1, 4, 56, 56}, torch::kFloat);
    torch::Tensor olist8_t = torch::from_blob(olist8, {1, 2, 9, 9}, torch::kFloat);
    torch::Tensor olist9_t = torch::from_blob(olist9, {1, 4, 9, 9}, torch::kFloat);
    torch::Tensor olist10_t = torch::from_blob(olist10, {1, 2, 5, 5}, torch::kFloat);
    torch::Tensor olist11_t = torch::from_blob(olist11, {1, 4, 5, 5}, torch::kFloat);

    std::vector<torch::Tensor> olist = {olist0_t, olist1_t, olist2_t, olist3_t, olist4_t, olist5_t,
                                        olist6_t, olist7_t, olist8_t, olist9_t, olist10_t, olist11_t};

    torch::Tensor bboxlist = postprocess(olist);
    *num_boxes = bboxlist.sizes()[0];

    return bboxlist.data_ptr<float>();
};





