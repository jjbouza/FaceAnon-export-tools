This folder sets up some infastructure for extracting the SOTA SFD face detector from the face-alignment package.

* `extract_onnx`: Extract the .onnx file for the underlying detector
* `processing`: C++ code emulating the preprocessing and postprocessing.
* `processing_torchscript`: Uses torchscript to compile the pre/post processing code. This is a (possibily temporary)
  replacement for the C++ processing code since iOS libtorch only exposes TorchScript at the moment.
* `face_alignment_native`: Python code for running face-alignment using C++ and ONNX backend, specifically for testing
  and verification that torchscript/ONNX models are equivalent to originals.

Here are the current strategies for porting the various components:
* `s3fd` face detector: Pre/post processing done in TorchScript modules. Actual network is ONNX/CoreML.
* `keypointrcnn`: Using a torchscript module possibily temporarily since ONNX exporting broken.
