/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSplitOp(A) is implemented using Stablehlo sliceOp
struct ONNXSplitOpLoweringToStablehlo : public ConversionPattern {
  ONNXSplitOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSplitOpAdaptor operandAdaptor(operands);
    ONNXSplitOp splitOp = mlir::dyn_cast<ONNXSplitOp>(op);
    Value input = splitOp.getInput();
    Value split = splitOp.getSplit();
    assert(isRankedShapedType(input.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType inputType = mlir::cast<ShapedType>(input.getType());
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    uint64_t rank = inputType.getRank();
    uint64_t outputNum = splitOp.getNumResults();
    int64_t dimIndex = splitOp.getAxis();
    if (dimIndex < 0)
      dimIndex += rank;
    int64_t inputDimSize = inputType.getDimSize(dimIndex);

    // Get a shape helper (not used?)
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXSplitOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    SmallVector<int64_t, 4> splitSizes;
    if (auto splitAttr = getElementAttributeFromONNXValue(split)) {
      for (IntegerAttr value : splitAttr.getValues<IntegerAttr>()) {
        int64_t splitSize = mlir::cast<IntegerAttr>(value).getInt();
        splitSizes.push_back(splitSize);
      }
    } else if (mlir::isa<NoneType>(split.getType())) {
      assert(!ShapedType::isDynamic(inputDimSize) &&
             "input dim size can't be dynamic");
      int64_t sliceSize = inputDimSize / outputNum;
      for (unsigned i = 0; i < outputNum; ++i)
        splitSizes.push_back(sliceSize);
    } else {
      assert(false && "dynamic split not yet supported");
    }

    // Check if input has dynamic dimensions
    bool hasDynamicDims = false;
    for (uint64_t i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(inputType.getDimSize(i))) {
        hasDynamicDims = true;
        break;
      }
    }

    SmallVector<int64_t, 4> sliceShape =
        llvm::to_vector<4>(inputType.getShape());
    SmallVector<Value, 4> slices;
    slices.reserve(outputNum);

    if (hasDynamicDims) {
      // Use real_dynamic_slice for inputs with dynamic dimensions
      MultiDialectBuilder<StablehloBuilder> create(rewriter, loc);
      
      int64_t beginIndice = 0;
      for (uint64_t i = 0; i < outputNum; ++i) {
        sliceShape[dimIndex] = splitSizes[i];
        Type sliceType =
            RankedTensorType::get(sliceShape, inputType.getElementType());
        
        // Build start_indices tensor
        SmallVector<Value> startIndicesVec;
        for (uint64_t d = 0; d < rank; ++d) {
          if ((int64_t)d == dimIndex) {
            Value startVal = create.stablehlo.constantI64(beginIndice);
            // Reshape scalar to rank-1 tensor for concatenation
            startVal = create.stablehlo.reshape(
                RankedTensorType::get({1}, rewriter.getI64Type()), startVal);
            startIndicesVec.push_back(startVal);
          } else {
            Value startVal = create.stablehlo.constantI64(0);
            // Reshape scalar to rank-1 tensor for concatenation
            startVal = create.stablehlo.reshape(
                RankedTensorType::get({1}, rewriter.getI64Type()), startVal);
            startIndicesVec.push_back(startVal);
          }
        }
        Value startIndices = rewriter.create<stablehlo::ConcatenateOp>(
            loc,
            RankedTensorType::get({static_cast<int64_t>(rank)}, 
                rewriter.getI64Type()),
            startIndicesVec, rewriter.getI64IntegerAttr(0));
        
        // Build limit_indices tensor
        SmallVector<Value> limitIndicesVec;
        for (uint64_t d = 0; d < rank; ++d) {
          if ((int64_t)d == dimIndex) {
            Value limitVal = create.stablehlo.constantI64(beginIndice + splitSizes[i]);
            // Reshape scalar to rank-1 tensor for concatenation
            limitVal = create.stablehlo.reshape(
                RankedTensorType::get({1}, rewriter.getI64Type()), limitVal);
            limitIndicesVec.push_back(limitVal);
          } else {
            // For dynamic dimensions, use stablehlo.get_dimension_size
            Value limitVal;
            if (ShapedType::isDynamic(inputType.getDimSize(d))) {
              Value dimSize = rewriter.create<stablehlo::GetDimensionSizeOp>(
                  loc, RankedTensorType::get({}, rewriter.getI32Type()),
                  input, rewriter.getI64IntegerAttr(d));
              // Convert i32 to i64
              dimSize = rewriter.create<stablehlo::ConvertOp>(
                  loc, RankedTensorType::get({}, rewriter.getI64Type()), 
                  dimSize);
              limitVal = dimSize;
            } else {
              limitVal = create.stablehlo.constantI64(inputType.getDimSize(d));
            }
            // Reshape scalar to rank-1 tensor for concatenation
            limitVal = create.stablehlo.reshape(
                RankedTensorType::get({1}, rewriter.getI64Type()), limitVal);
            limitIndicesVec.push_back(limitVal);
          }
        }
        Value limitIndices = rewriter.create<stablehlo::ConcatenateOp>(
            loc,
            RankedTensorType::get({static_cast<int64_t>(rank)}, 
                rewriter.getI64Type()),
            limitIndicesVec, rewriter.getI64IntegerAttr(0));
        
        // Build strides tensor (all ones)
        SmallVector<Value> stridesVec;
        for (uint64_t d = 0; d < rank; ++d) {
          Value strideVal = create.stablehlo.constantI64(1);
          // Reshape scalar to rank-1 tensor for concatenation
          strideVal = create.stablehlo.reshape(
              RankedTensorType::get({1}, rewriter.getI64Type()), strideVal);
          stridesVec.push_back(strideVal);
        }
        Value strides = rewriter.create<stablehlo::ConcatenateOp>(
            loc,
            RankedTensorType::get({static_cast<int64_t>(rank)}, 
                rewriter.getI64Type()),
            stridesVec, rewriter.getI64IntegerAttr(0));
        
        slices.push_back(create.stablehlo.real_dynamic_slice(
            sliceType, input, startIndices, limitIndices, strides));
        
        beginIndice += splitSizes[i];
      }
    } else {
      // Use static slice for inputs without dynamic dimensions
      SmallVector<int64_t, 4> beginIndices(rank, 0);
      SmallVector<int64_t, 4> endIndices =
          llvm::to_vector<4>(inputType.getShape());
      SmallVector<int64_t, 4> strides(rank, 1);
      
      int64_t beginIndice = 0;
      int64_t endIndice = 0;
      for (uint64_t i = 0; i < outputNum; ++i) {
        sliceShape[dimIndex] = splitSizes[i];
        Type sliceType =
            RankedTensorType::get(sliceShape, inputType.getElementType());
        endIndice += splitSizes[i];
        beginIndices[dimIndex] = beginIndice;
        endIndices[dimIndex] = endIndice;
        slices.push_back(stablehlo::SliceOp::create(rewriter, loc, sliceType,
            input, DenseI64ArrayAttr::get(context, beginIndices),
            DenseI64ArrayAttr::get(context, endIndices),
            DenseI64ArrayAttr::get(context, strides)));
        beginIndice = endIndice;
      }
    }
    
    rewriter.replaceOp(op, slices);
    return success();
  }
};

} // namespace

void populateLoweringONNXSplitOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
