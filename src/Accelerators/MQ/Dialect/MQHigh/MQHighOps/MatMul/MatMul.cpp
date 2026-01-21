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

  // LHS: [M_tiles, K_tiles, tile_m, tile_k]
  // RHS: [K_tiles, N_tiles, tile_k, tile_n]
  int64_t m_tiles = lhsType.getDimSize(0);
  int64_t n_tiles = rhsType.getDimSize(1);
  int64_t tile_m  = lhsType.getDimSize(2);
  int64_t tile_n  = rhsType.getDimSize(3);
  llvm::outs() << "[MQHighMatMulOp] m_tiles: " << m_tiles << ", n_tiles: " << n_tiles
               << ", tile_m: " << tile_m << ", tile_n: " << tile_n << "\n";

  SmallVector<int64_t, 4> resultShape = {m_tiles, n_tiles, tile_m, tile_n};
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));

  return success();
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult MQHighMatMulOp::verify() {
  MQHighMatMulOpAdaptor adaptor(*this);
  
  auto lhsType = mlir::dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhsType = mlir::dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
  auto resType = mlir::dyn_cast<RankedTensorType>(getResult().getType());

  if (!lhsType || !rhsType || !resType) {
    llvm::outs() << "MQ Error: [Verifier] All operands must be ranked tensors.\n";
    return failure();
  }

  // 1. 检查 Rank 是否为 4
  if (lhsType.getRank() != 4 || rhsType.getRank() != 4 || resType.getRank() != 4) {
    llvm::outs() << "MQ Error: [Verifier] Operands must be Rank 4 (tiled format).\n";
    return failure();
  }

  auto checkDim = [](int64_t d1, int64_t d2, const char* msg) {
    if (mlir::ShapedType::isDynamic(d1) || mlir::ShapedType::isDynamic(d2)) return true; 
    if (d1 != d2) {
      llvm::outs() << "MQ Error: [Verifier] " << msg << " (" << d1 << " vs " << d2 << ")\n";
      return false;
    }
    return true;
  };

  // 2. K 维度校验 (LHS 维 1&3 vs RHS 维 0&2)
  // LHS: [M_t, K_t, 16, tile_k]  RHS: [K_t, N_t, tile_k, tile_n]
  if (!checkDim(lhsType.getDimSize(1), rhsType.getDimSize(0), "K_tiles mismatch"))
    return failure();
  if (!checkDim(lhsType.getDimSize(3), rhsType.getDimSize(2), "tile_k inner size mismatch"))
    return failure();

  // 3. M 维度校验 (Result vs LHS)
  if (!checkDim(resType.getDimSize(0), lhsType.getDimSize(0), "M_tiles mismatch"))
    return failure();
  if (!checkDim(resType.getDimSize(2), lhsType.getDimSize(2), "tile_m inner size mismatch"))
    return failure();

  // 4. N 维度校验 (Result vs RHS)
  if (!checkDim(resType.getDimSize(1), rhsType.getDimSize(1), "N_tiles mismatch"))
    return failure();
  if (!checkDim(resType.getDimSize(3), rhsType.getDimSize(3), "tile_n inner size mismatch"))
    return failure();

  // 5. Bias 校验 (MVP 阶段如果是 Optional 且为空则跳过)
  if (Value bias = adaptor.getBias()) {
    auto biasType = mlir::dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType) return failure();
    // TODO: 根据硬件需求定义 4D Bias 的校验逻辑
  }

  return success();
}

} // namespace mqhigh
} // namespace onnx_mlir
