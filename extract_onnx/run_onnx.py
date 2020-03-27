import onnxruntime
import numpy as np

from skimage import io


def run_onnx_random():
    sess = onnxruntime.InferenceSession("sfd_detector.onnx")
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_name = sess.get_inputs()[0].name

    #Random input
    img = np.random.random(input_shape).astype(np.float32)
    result = sess.run([], {input_name: img})    

    return result

def run_onnx_image(fname):
    sess = onnxruntime.InferenceSession("sfd_detector.onnx")
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_name = sess.get_inputs()[0].name

    #Random input
    img = io.imread(fname).transpose(2,0,1).reshape(input_shape).astype(np.float32)
    result = sess.run([], {input_name: img})    
    return result

run_onnx_random()
result = run_onnx_image("aflw-test.jpg")
print(result)
