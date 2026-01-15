/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MQHighOps.cpp - MQHigh Operations --------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the MQHigh operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace mqhigh {

//===----------------------------------------------------------------------===//
// MQHighDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void MQHighDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.cpp.inc"
      >();
}

} // namespace mqhigh
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
// Keep this part at the end of the file.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.cpp.inc"

// TODO
// #define GET_ATTRDEF_CLASSES
// #include "src/Accelerators/MQ/Dialect/MQHigh/MQHighAttributes.cpp.inc"

#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighDialect.cpp.inc"
