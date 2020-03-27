// for testing.

#include <iostream>
#include "processing.h"

#include <torch/torch.h>

int main(){

    torch::Tensor img = torch::randn({450, 450, 3});

    // test preprocessing
    std::cout << "Testing Preprocessing..." << std::endl;
    torch::Tensor img_preprocessed = preprocess(img);

    std::cout << "Original image size:" << img.sizes() << std::endl;
    std::cout << "Preprocessed size:" << img_preprocessed.sizes() << std::endl;

    // test postprocessing
    std::vector<torch::Tensor> olist = {torch::randn({1,2,112,112}), torch::randn({1,4,112,112}), 
                                        torch::randn({1,2,56,56}), torch::randn({1,4,56,56}),
                                        torch::randn({1,2,56,56}), torch::randn({1,4,56,56}),
                                        torch::randn({1,2,56,56}), torch::randn({1,4,56,56}),
                                        torch::randn({1,2,9,9}), torch::randn({1,4,9,9}),
                                        torch::randn({1,2,5,5}), torch::randn({1,4,5,5})};

    torch::Tensor output = postprocess(olist);
    std::cout << output.sizes() << std::endl;

    std::cout << "Done!" << std::endl;

    return 0;
}
