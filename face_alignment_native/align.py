import ctypes

import onnxruntime
from skimage import io
import numpy as np


lib_loc = "/home/josebouza/Projects/face-bounding-box-iOS/processing/build/libprocessing.so"
processing = ctypes.CDLL(lib_loc)

def run_align(fname):
    img = io.imread(fname).astype(np.float32)
    print(img.shape)
    img_c = img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    img_pre = processing.preprocess_wrapper(img_c)
    print(img.shape)



result = run_align("img.jpg")

