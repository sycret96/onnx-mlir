/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MatMul.cpp - MQHigh Operations ---------------------===//
//
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

// #include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps/ShapeHelper.hpp"
// #include "mlir/IR/BuiltinTypes.h"

#include "mlir/Support/LLVM.h"
#include "src/Support/Diagnostic.hpp"
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace mqhigh {

// TODO
//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//
LogicalResult MQHighMatMulOp::inferReturnTypes(
    MLIRContext *context, 
    ::std::optional<Location> location,
    ValueRange operands, 
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions, 
    SmallVectorImpl<Type> &inferredReturnTypes) {

  auto lhsType = dyn_cast_or_null<RankedTensorType>(operands[0].getType());
  auto rhsType = dyn_cast_or_null<RankedTensorType>(operands[1].getType());
  auto elementType = lhsType.getElementType();

  // Shape [M, K] @ [K, N]
  int64_t m = lhsType.getShape()[0];
  int64_t n = rhsType.getShape()[1];

  SmallVector<int64_t, 2> resultShape = {m, n};
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));

  return success();
}

LogicalResult MQHighMatMulOp::verify() {
  MQHighMatMulOpAdaptor adaptor(*this);
  
  auto lhsType = llvm::dyn_cast_or_null<RankedTensorType>(adaptor.getLhs().getType());
  auto rhsType = llvm::dyn_cast_or_null<RankedTensorType>(adaptor.getRhs().getType());
  auto resType = llvm::dyn_cast_or_null<RankedTensorType>(getResult().getType());

  if (!lhsType || !rhsType || !resType) {
    llvm::outs() << "[Verifier] Error: Requires ranked tensor types.\n";
    return failure();
  }

  // TODO: 2D matmul only for now
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || resType.getRank() != 2) {
    llvm::outs() << "[Verifier] Error: All operands must be Rank 2.\n";
    return failure();
  }

  auto checkDim = [](int64_t d1, int64_t d2, const char* msg) {
    // 关键：只要有一个是动态的 (kDynamic)，就跳过检查
    if (mlir::ShapedType::isDynamic(d1) || mlir::ShapedType::isDynamic(d2)) {
      return true; 
    }
    if (d1 != d2) {
      llvm::outs() << "[Verifier] Error: " << msg << " (" << d1 << " vs " << d2 << ")\n";
      return false;
    }
    return true;
  };

  if (!checkDim(lhsType.getDimSize(1), rhsType.getDimSize(0), "K dimension mismatch"))
    return failure();
  if (!checkDim(resType.getDimSize(0), lhsType.getDimSize(0), "M dimension mismatch"))
    return failure();
  if (!checkDim(resType.getDimSize(1), rhsType.getDimSize(1), "N dimension mismatch"))
    return failure();

  if (Value bias = adaptor.getBias()) {
    auto biasType = llvm::dyn_cast_or_null<RankedTensorType>(bias.getType());
    if (!biasType) return failure();

    int64_t rank = biasType.getRank();
    if (rank == 1) {
      if (!checkDim(biasType.getDimSize(0), resType.getDimSize(1), "1D bias mismatch"))
        return failure();
    } else if (rank == 2) {
      if (!checkDim(biasType.getDimSize(0), resType.getDimSize(0), "2D bias M mismatch") ||
          !checkDim(biasType.getDimSize(1), resType.getDimSize(1), "2D bias N mismatch"))
        return failure();
    }
  }

  return success();
}

} // namespace mqhigh
} // namespace onnx_mlir
