/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- MQCompilerUtils.cpp -----------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Utilities for MQ Accelerator
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
// #include "llvm/CodeGen/TargetPassConfig.h"
// #include "llvm/IR/DataLayout.h"
// #include "llvm/MC/TargetRegistry.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/Support/TargetSelect.h"
// #include "llvm/Target/TargetMachine.h"

// #include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/MQ/Compiler/MQCompilerUtils.hpp"
#include "src/Accelerators/MQ/Pass/MQPasses.hpp"
// #include "src/Accelerators/NNPA/Compiler/ZHighDisposableGarbageCollector.hpp"
#include "src/Accelerators/MQ/Dialect/MQHigh/MQHighOps.hpp"
// #include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
// #include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
// #include "src/Accelerators/NNPA/Support/NNPALimit.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "MQCompilerUtils"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

void configurePassesMQ() {
  // Placeholder: Configure options specific to MQ accelerator here.
  // For example, setting up quantization options or hardware limits.
}

void addONNXToMQHighPasses(mlir::PassManager &pm) {
  // 1. Pre-processing: Simplify shape related ops to help with shape inference
  // Run this loop to propagate constants and helper functions
  for (unsigned i = 0; i < 3; i++) {
    // pm.addPass(onnx_mlir::createRewriteONNXForMQHighPass()); // Future: Custom rewrites
    pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  }

  // 2. Core Lowering: ONNX -> MQHigh
  // This Pass does not exist yet. You need to implement it in src/Accelerators/MQ/Conversion
  pm.addPass(onnx_mlir::createONNXToMQHighPass());
  
  // 3. Shape Inference: Essential after dialect conversion
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());

  // 4. Post-lowering optimizations
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  
  // 5. Constant Propagation (Optional but recommended)
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
}


/*
不同level的emit枚举从高层到低层依次为（枚举类型值从小到大）
  EmitONNXBasic,
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitObj,
  EmitLib,
  EmitJNI,
*/
void addPassesMQ(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) {
  
  // 1. Configure Global Options
  configurePasses();
  configurePassesMQ();

  // 2. Input to MLIR (High Level)
  if (emissionTarget >= EmitMLIR) {
    // Lower ONNX ops to MQHigh dialect where possible
    addONNXToMQHighPasses(pm);

    // If we only want high-level IR, we can stop here or proceed to lower everything else
    pm.addPass(mlir::createCanonicalizerPass());
  }
}

} // namespace onnx_mlir
