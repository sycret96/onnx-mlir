// SPDX-License-Identifier: Apache-2.0

//===- ONNXToMQHigh.td - Replacing ONNX Ops by MQHigh Ops -*- tablegen ----===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_TO_MQHIGH
#define ONNX_TO_MQHIGH

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/MQ/Dialect/MQHigh/MQHigh.td"
#endif // OP_BASE

// Place simple tablegen patterns here as you add ops.
// See src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.td for examples.

#endif // ONNX_TO_MQHIGH
