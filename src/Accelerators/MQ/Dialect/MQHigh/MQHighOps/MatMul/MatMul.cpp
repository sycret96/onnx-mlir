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
#include "src/Support/Diagnostic.hpp"
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp"

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

LogicalResult MQHighMatMulOp::verify() {
  MQHighMatMulOpAdaptor adaptor(*this);
  
  auto lhsType = mlir::dyn_cast_or_null<RankedTensorType>(adaptor.getLhs().getType());
  auto rhsType = mlir::dyn_cast_or_null<RankedTensorType>(adaptor.getRhs().getType());
  auto resType = mlir::dyn_cast_or_null<RankedTensorType>(getResult().getType());

  // 1. Basic Type Check
  if (!lhsType || !rhsType || !resType) return failure();

  // 2. Get Dimensions (Static Shape Only)
  int64_t lhs_M = lhsType.getDimSize(0);
  int64_t lhs_K = lhsType.getDimSize(1);
  int64_t rhs_K = rhsType.getDimSize(0);
  int64_t rhs_N = rhsType.getDimSize(1);
  int64_t res_M = resType.getDimSize(0);
  int64_t res_N = resType.getDimSize(1);

  // 3. Verify MatMul Shape: [M, K] * [K, N] = [M, N]
  if (lhs_K != rhs_K) return failure();
  if (res_M != lhs_M) return failure();
  if (res_N != rhs_N) return failure();

  // 4. Optional Bias Check
  if (Value bias = adaptor.getBias()) {
    auto biasType = mlir::dyn_cast_or_null<RankedTensorType>(bias.getType());
    if (!biasType) return failure();

    int64_t biasRank = biasType.getRank();
    
    // Check 1D Bias: [N]
    if (biasRank == 1) {
      if (biasType.getDimSize(0) != res_N) return failure();
    } 
    // Check 2D Bias: [M, N]
    else if (biasRank == 2) {
      if (biasType.getDimSize(0) != res_M || biasType.getDimSize(1) != res_N) {
        return failure();
      }
    } 
    else {
      return failure();
    }
  }

  return success();
}

} // namespace mqhigh
} // namespace onnx_mlir
