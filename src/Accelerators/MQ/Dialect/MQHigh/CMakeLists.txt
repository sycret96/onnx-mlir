add_subdirectory(MQHighOps)

add_onnx_mlir_dialect(MQHigh mqhigh)
add_onnx_mlir_dialect_doc(mqhigh MQHigh.td)

add_onnx_mlir_library(OMMQHighOps
  # Top files for MQHigh dialect.
  MQHighOps.cpp
#   MQHighOps/OpHelper.cpp # TODO

  # Support for shape inference, verifier, and canonicalization.
  # Please list in the alphabetical order.
#   MQHighOps/Conv2D/Conv2D.cpp
#   MQHighOps/Elementwise/Elementwise.cpp
  MQHighOps/MatMul/MatMul.cpp
  MQHighOps/Pack/Pack.cpp
  MQHighOps/Unpack/Unpack.cpp
#   MQHighOps/Pooling/Pooling.cpp
#   MQHighOps/Reshape/Reshape.cpp
#   MQHighOps/Softmax/Softmax.cpp

  DEPENDS
#   OMLayoutHelper # TODO 这是什么？
  OMMQHighIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
#   OMLayoutHelper # TODO 这是什么？
  OMShapeHelperOpInterface
  OMMQCompilerOptions
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${MQ_INCLUDE_PATH}
  )