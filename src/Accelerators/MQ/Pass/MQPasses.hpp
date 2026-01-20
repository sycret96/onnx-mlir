/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- MQPasses.hpp - MQ Passes Definition -----------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for MQ in
// addition to the passes used by ONNX MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_MQ_PASSES_H
#define ONNX_MLIR_MQ_PASSES_H

#include "mlir/Pass/Pass.h"

// Include your custom compiler options if needed
// #include "src/Accelerators/MQ/Compiler/MQCompilerOptions.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// MQHigh Passes
//===----------------------------------------------------------------------===//

/// Add pass for lowering ONNX ops to MQHigh ops.
std::unique_ptr<mlir::Pass> createONNXToMQHighPass();
// void configureONNXToZHighLoweringPass(bool reportOnNNPAUnsupportedOps,
//     bool isDynQuant, bool quantIsActivationSym, bool quantIsWeightSym,
//     llvm::ArrayRef<std::string> quantOpTypes);

// TODO: Add pass for rewriting ONNX ops for MQHigh.
// std::unique_ptr<mlir::Pass> createRewriteONNXForMQHighPass();

// TODO: Add pass for re-construct ONNX ops from MQHigh ops.
// std::unique_ptr<mlir::Pass> createMQHighToONNXPass();


namespace mqhigh {

// TODO: Pass for layout propagation at MQHighIR.
// std::unique_ptr<mlir::Pass> createMQHighLayoutPropagationPass();

// TODO: Pass for constant propagation at MQHighIR.
// std::unique_ptr<mlir::Pass> createMQHighConstPropagationPass();

} // end namespace mqhigh


//===----------------------------------------------------------------------===//
// MQLow Passes
//===----------------------------------------------------------------------===//

namespace mqlow {

// TODO: Add pass for rewriting MQLow ops.
// std::unique_ptr<mlir::Pass> createMQLowRewritePass();

} // namespace mqlow
} // namespace onnx_mlir
#endif
