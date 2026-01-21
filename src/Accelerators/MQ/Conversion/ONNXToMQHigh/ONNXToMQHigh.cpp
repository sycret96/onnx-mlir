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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp"
#include "src/Accelerators/MQ/Pass/MQPasses.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX to MQHigh Converson Patterns
//===----------------------------------------------------------------------===//
struct ONNXMatMulOpPattern : public OpConversionPattern<ONNXMatMulOp> {
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // 1. 获取输入类型
    auto typeA = mlir::cast<RankedTensorType>(adaptor.getA().getType());
    auto typeB = mlir::cast<RankedTensorType>(adaptor.getB().getType());
    auto elementType = typeA.getElementType();

    // 2. 根据数据类型确定分块参数
    int64_t n_val = 1;
    if (elementType.isBF16() || elementType.isF16()) n_val = 2;
    else if (elementType.isInteger(8)) n_val = 4;

    int64_t tile_m = 16;
    int64_t tile_k = 4 * n_val;
    int64_t tile_n = 64;

    // 3. 计算分块后的 Tensor 形状
    SmallVector<int64_t> shapeAPacked = {
        typeA.getDimSize(0) / tile_m, typeA.getDimSize(1) / tile_k, tile_m, tile_k};
    auto typeAPacked = RankedTensorType::get(shapeAPacked, elementType);

    SmallVector<int64_t> shapeBPacked = {
        typeB.getDimSize(0) / tile_k, typeB.getDimSize(1) / tile_n, tile_k, tile_n};
    auto typeBPacked = RankedTensorType::get(shapeBPacked, elementType);

    // 4. 使用 OpTy::create 替代 rewriter.create
    // 插入 Pack A (Z_Z 布局)
    auto innerTilesA = rewriter.getI64ArrayAttr({tile_m, tile_k});
    auto packA = mqhigh::MQHighPackOp::create(
        rewriter, loc, typeAPacked, adaptor.getA(), innerTilesA, rewriter.getStringAttr("Z_Z"));

    // 插入 Pack B (N_N 布局)
    auto innerTilesB = rewriter.getI64ArrayAttr({tile_k, tile_n});
    auto packB = mqhigh::MQHighPackOp::create(
        rewriter, loc, typeBPacked, adaptor.getB(), innerTilesB, rewriter.getStringAttr("N_N"));

    // 5. 创建 MQHighMatMulOp，传入 4D 类型
    SmallVector<int64_t, 4> matmulShape = {
      typeA.getDimSize(0) / tile_m,
      typeB.getDimSize(1) / tile_n, 
      tile_m,
      tile_n
    };
    auto matmulResType = RankedTensorType::get(matmulShape, elementType);
    auto resMatMul = mqhigh::MQHighMatMulOp::create(
      rewriter, loc, matmulResType, packA.getResult(), packB.getResult());

    // 6. 插入 Unpack 操作，将 4D 输出转换回原始 2D 形状
    auto unpackOp = mqhigh::MQHighUnpackOp::create(
    rewriter, loc, 
    op.getType(),           // 原始 2D Tensor 类型 (M x N)
    resMatMul.getResult(),  // MatMul 的 4D 输出
    rewriter.getI64ArrayAttr({tile_m, tile_n}), // 当初 Pack 到输出端的分块大小
    rewriter.getStringAttr("Z_Z")               // 保持布局属性一致
);

    // 7. 替换原始 OP，使用 unpack 的结果
    rewriter.replaceOp(op, unpackOp.getOutput());

    return success();
  }
};


void getONNXToMQHighOneOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ONNXMatMulOpPattern>(context);
}

//===----------------------------------------------------------------------===//
// ONNX to MQHigh Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct ONNXToMQHighLoweringPass
    : public PassWrapper<ONNXToMQHighLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToMQHighLoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-mqhigh"; }

  StringRef getDescription() const override {
    return "Lower ONNX ops to MQHigh ops.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace

void ONNXToMQHighLoweringPass::runOnOperation() {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());

    // Define LEGAL dialects
    target.addLegalDialect<mqhigh::MQHighDialect,
        ONNXDialect,
        func::FuncDialect,
        arith::ArithDialect>();

    // TODO: fusion onnx.matmul + onnx.add -> mqhigh.matmul
    // RewritePatternSet combinedPatterns(&getContext());
    // onnx_mlir::getONNXToMQHighMultipleOpPatterns(combinedPatterns);
    // (void)applyPatternsGreedily(module, std::move(combinedPatterns));    // It's ok to fail.

    // 这不知道是干啥的 看起来是官方给的接口 可以直接用吗？
    // Run the unknown dimension analysis to help check equality of unknown
    // dimensions at compile time.
    onnx_mlir::DimAnalysis dimAnalysis(module);
    dimAnalysis.analyze();

    // Single ONNX to MQHigh operation lowering.
    RewritePatternSet patterns(&getContext());
    onnx_mlir::getONNXToMQHighOneOpPatterns(patterns);
    llvm::outs() << "getONNXToMQHighOneOpPatterns done.\n";

    // This is to make sure we don't want to alloc any MemRef at this high-level
    // representation.
    target.addIllegalOp<mlir::memref::AllocOp>();
    target.addIllegalOp<mlir::memref::DeallocOp>();
    target.addDynamicallyLegalOp<ONNXMatMulOp>([](ONNXMatMulOp op) {
      return false;
    });

    // TODO: ONNX ops to MQHigh dialect under specific conditions.
    // When adding a new op, need to implement a method,
    // for the op in ONNXLegalityCheck.cpp.
    // getONNXToMQHighOneOpDynamicallyLegal(&target, &dimAnalysis);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }


std::unique_ptr<Pass> createONNXToMQHighPass() {
  return std::make_unique<ONNXToMQHighLoweringPass>();
}

} // namespace onnx_mlir
