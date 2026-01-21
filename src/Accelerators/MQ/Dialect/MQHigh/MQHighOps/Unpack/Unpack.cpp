/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Unpack.cpp - MQHigh Operations ---------------------===//
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
LogicalResult MQHighUnpackOp::verify() {   
    // TODO: add verification logic 
    return success();
}
} // namespace mqhigh
} // namespace onnx_mlir
