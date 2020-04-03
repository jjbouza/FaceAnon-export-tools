from onnx_coreml import convert

mlmodel = convert(model='./sfd_detector.onnx', minimum_ios_deployment_target='13')
mlmodel.save('./sfd_detector.mlmodel')

# Not working:
#mlmodel = convert(model='./keypointrcnn.onnx', minimum_ios_deployment_target='13')
#mlmodel.save('./keypointrcnn.mlmodel')
