# SPDX-License-Identifier: Apache-2.0

# 1. 启用开关与作用域上报
# 按照 onnx-mlir 规范，通过 PARENT_SCOPE 告知主构建系统该加速器已激活
set(MQ_ENABLED 1 BOOL PARENT_SCOPE)

# 2. 定义根路径变量 (方便后续引用)
set(MQ_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(MQ_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

# 3. 设置输出路径（继承自全局配置）
set(MQ_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(MQ_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(MQ_INCLUDE_PATH ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

# 4. 引用 ONNX-MLIR 的核心路径
set(MQ_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
set(MQ_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})

# 5. 外部依赖检测 (例如自研的 RISC-V 运行库或 SDK)
# 仿照 zDNN 的逻辑，这里假设有一个 MQ_SDK_DIR
if (DEFINED ENV{MQ_SDK_DIR} AND EXISTS $ENV{MQ_SDK_DIR})
  message(STATUS "MQ SDK Found: " $ENV{MQ_SDK_DIR})
  set(MQ_SDK_DIR $ENV{MQ_SDK_DIR})
  set(MQ_SDK_ENABLED 1)
else()
  message(WARNING "MQ_SDK_DIR not found. Some hardware tests might fail.")
  set(MQ_SDK_ENABLED 0)
endif()

# 如果有专门的依赖配置脚本，可以在此引入
# include(MQ_deps.cmake)

# 6. 编译选项配置 (Debug / 特性开关)
option(MQ_DEBUG "Enable MQ debug information" OFF)
if (MQ_DEBUG)
  add_compile_definitions(MQ_DEBUG)
endif()

# 7. 包含子目录 (MLIR 必须的模块)
# TODO 逐个完成
add_subdirectory(Dialect)      # 定义 MQ Dialect
# add_subdirectory(Conversion)   # 定义 ONNX -> MQ 的转换
# add_subdirectory(Support)      # 辅助工具
# add_subdirectory(Transform)    # 针对 MQ 的 Pass
add_subdirectory(Compiler)     # 编译插件逻辑
# add_subdirectory(Runtime)      # 运行时支持

# 8. 定义目标库 (核心目标)
add_onnx_mlir_library(OMMQAccel
  MQAccelerator.cpp     # 加速器接口实现文件

  # 插件规范：不直接打包进主库，按需链接
  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  # 依赖的子模块库名，通常在上述子目录的 CMakeLists 中定义
  # OMMQRuntime
  # libmq_sdk (如果是外部库)

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${MQ_ONNX_MLIR_SRC_ROOT}
  ${MQ_SRC_ROOT}
  ${MQ_BIN_ROOT}
  ${MQ_INCLUDE_PATH}
  # 如果有 SDK 头文件，也加在这里
  ${MQ_SDK_DIR}/include

  LINK_LIBS PUBLIC
  onnx
  OMAccelerator                # 必须链接，提供 Accelerator 基类
  OMMQCompilerUtils     # 在 Compiler 子目录定义的库
  OMMQHighOps                 # 在 Dialect 子目录定义的库
#   OMMQLowOps

#   根据转换路径添加对应的库 
#   TODO
#   OMMQHighToMQLow
#   OMMQLowToLLVM
)