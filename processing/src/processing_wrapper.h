#include "processing.h"

extern "C"{
float* preprocess_wrapper(float* img);
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
                           int* num_boxes);
}
