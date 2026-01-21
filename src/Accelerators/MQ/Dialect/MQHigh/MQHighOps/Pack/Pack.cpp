/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pack.cpp - MQHigh Operations ---------------------===//
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

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//
LogicalResult MQHighPackOp::verify() {
    auto inputShape = getInput().getType().getShape();
    auto outputShape = getOutput().getType().getShape();
    auto tiles = getInnerTileSizes(); 

    // 1. Check if the number of tile sizes matches the input rank
    if (inputShape.size() != tiles.size()) {
        llvm::outs() << "MQ Error: The number of inner_tile_sizes must match the rank of the input tensor.\n";
        return failure(); 
    }

    // 2. Mathematical consistency check
    for (size_t i = 0; i < inputShape.size(); ++i) {
        int64_t tileSize = mlir::cast<IntegerAttr>(tiles[i]).getInt();
        
        // Static divisibility check
        if (!ShapedType::isDynamic(inputShape[i]) && inputShape[i] % tileSize != 0) {
            llvm::outs() << "MQ Error: Dimension " << i << " (size: " << inputShape[i] 
                         << ") is not divisible by tile size " << tileSize << ".\n";
            return failure();
        }

        // Verify output outer dimensions (M/tile)
        if (outputShape[i] != inputShape[i] / tileSize) {
            llvm::outs() << "MQ Error: Output outer dimension " << i << " mismatch. "
                         << "Expected: " << (inputShape[i] / tileSize) 
                         << ", Got: " << outputShape[i] << ".\n";
            return failure();
        }

        // Verify output inner dimensions (tile)
        // Note: The inner dimensions are typically appended after the outer ones
        if (outputShape[i + inputShape.size()] != tileSize) {
            llvm::outs() << "MQ Error: Output inner dimension " << i << " mismatch. "
                         << "Expected: " << tileSize 
                         << ", Got: " << outputShape[i + inputShape.size()] << ".\n";
            return failure();
        }
    }
    
    return success();
}
} // namespace mqhigh
} // namespace onnx_mlir
