add_onnx_mlir_library(OMMQCompilerOptions
  MQCompilerOptions.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${MQ_SRC_ROOT}
  ${MQ_BIN_ROOT}
  ${MQ_ONNX_MLIR_SRC_ROOT}
  ${MQ_ONNX_MLIR_BIN_ROOT}

  LINK_LIBS PUBLIC
  OMCompilerOptions

  ACCEL_INCLUDE_DIRS PRIVATE
  ${MQ_ONNX_MLIR_SRC_ROOT}
  ${MQ_ONNX_MLIR_BIN_ROOT}
  )
add_onnx_mlir_library(OMMQCompilerUtils
  MQCompilerUtils.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${MQ_SRC_ROOT}
  ${MQ_BIN_ROOT}
  ${MQ_ONNX_MLIR_SRC_ROOT}
  ${MQ_ONNX_MLIR_BIN_ROOT}

  LINK_LIBS PUBLIC
  OMMQCompilerOptions
  OMCompilerPasses
  OMMQToMQHigh     # 新增：让 Compiler 需要链接转换Pass库

  ACCEL_INCLUDE_DIRS PRIVATE
  ${MQ_ONNX_MLIR_SRC_ROOT}
  ${MQ_ONNX_MLIR_BIN_ROOT}
  )
