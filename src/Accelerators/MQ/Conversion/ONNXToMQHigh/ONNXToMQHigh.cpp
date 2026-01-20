/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToMQHigh.cpp - ONNX dialect to MQHigh lowering -----------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and MQHigh operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/MQ/Conversion/ONNXToMQHigh/ONNXToMQHigh.hpp"
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp"
#include "src/Accelerators/MQ/Pass/MQPasses.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX to MQHigh Converson Patterns
//===----------------------------------------------------------------------===//

// Example Pattern: Convert ONNXMatMul directly to MQHighMatMul
// This is the simplest MVP lowering without sticking/unsticking logic
struct ONNXMatMulOpPattern : public OpConversionPattern<ONNXMatMulOp> {
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO: add layout conversion here.
    
    mlir::Value noneValue;
    rewriter.replaceOpWithNewOp<mqhigh::MQHighMatMulOp>(
        op, op.getType(), adaptor.getA(), adaptor.getB(), noneValue);
    
    return success();
  }
};


void getONNXToMQHighOneOpPatterns(RewritePatternSet &patterns) {
  patterns.insert<ONNXMatMulOpPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ONNX to MQHigh Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct ONNXToMQHighLoweringPass
    : public PassWrapper<ONNXToMQHighLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToMQHighLoweringPass)

  StringRef getArgument() const override { return "onnx-to-mqhigh"; }

  StringRef getDescription() const override {
    return "Lower ONNX ops to MQHigh ops.";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());

    // Define LEGAL dialects
    target.addLegalDialect<mqhigh::MQHighDialect>();
    target.addLegalDialect<ONNXDialect>(); // Keep ONNX legal for unsupported ops

    // Define what needs to be converted
    target.addDynamicallyLegalOp<ONNXMatMulOp>([](ONNXMatMulOp op) {
      // If we supported it, it's illegal (must be converted).
      // If we didn't support it, it stays legal.
      return false; // For MVP, we force convert all MatMuls
    });

    RewritePatternSet patterns(&getContext());
    getONNXToMQHighOneOpPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createONNXToMQHighPass() {
  return std::make_unique<ONNXToMQHighLoweringPass>();
}

} // namespace onnx_mlir
