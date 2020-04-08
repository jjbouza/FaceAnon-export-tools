from onnx_coreml import convert
import coremltools
from coremltools.models.neural_network import flexible_shape_utils


mlmodel = convert(model='./sfd_detector.onnx', minimum_ios_deployment_target='13')
spec = mlmodel.get_spec()
spec.description.input[0].type.multiArrayType.shape[2] = 450
spec.description.input[0].type.multiArrayType.shape[3] = 450
flexible_shape_utils.set_multiarray_ndshape_range(spec, feature_name='input_img', lower_bounds=[1,3,1,1],
                                                  upper_bounds=[1,3,-1,-1])
mlmodel = coremltools.models.MLModel(spec)
mlmodel.save('./sfd_detector.mlmodel')

