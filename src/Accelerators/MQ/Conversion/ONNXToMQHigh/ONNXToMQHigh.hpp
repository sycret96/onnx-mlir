/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToMQHigh.hpp - ONNX dialect to MQHigh lowering -----------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and MQHigh operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TO_MQHIGH_H
#define ONNX_MLIR_ONNX_TO_MQHIGH_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

// Exports ONNXtoMQHigh patterns.
void getONNXToMQHighOneOpPatterns(mlir::RewritePatternSet &patterns);

// Exports ONNXtoMQHigh dynamically legal checks.
void getONNXToMQHighOneOpDynamicallyLegal(
    mlir::ConversionTarget *target, const DimAnalysis *dimAnalysis);

} // namespace onnx_mlir
#endif
