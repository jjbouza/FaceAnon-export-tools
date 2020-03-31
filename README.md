This folder sets up some infastructure for extracting the SOTA SFD face detector from the face-alignment package.

* `extract_onnx`: Extract the .onnx file for the underlying detector
* `processing`: C++ code emulating the preprocessing and postprocessing.
* `processing_torchscript`: Uses torchscript to compile the pre/post processing code. This is a (possibily temporary)
  replacement for the C++ processing code since iOS libtorch only exposes TorchScript at the moment.
* `face_alignment_native`: Python code for running face-alignment using C++ and ONNX backend.
