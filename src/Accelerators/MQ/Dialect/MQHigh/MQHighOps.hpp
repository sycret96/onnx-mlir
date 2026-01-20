/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MQHighOps.hpp - MQHigh Operations -------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the MQHigh operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_MQHIGH_H
#define ONNX_MLIR_MQHIGH_H

#include <map>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

namespace mlir {
// Traits can be added here if needed in the future
} // namespace mlir

/// Include the auto-generated header files containing the declarations of the
/// MQHigh dialect and operations.
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighDialect.hpp.inc"

// TODO
// #define GET_ATTRDEF_CLASSES
// #include "src/Accelerators/MQ/Dialect/MQHigh/MQHighAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp.inc"

#endif
