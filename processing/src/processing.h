#include <torch/torch.h>

#include <vector>

torch::Tensor preprocess(torch::Tensor img);
torch::Tensor postprocess(std::vector<torch::Tensor> olist);
