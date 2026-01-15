// SPDX-License-Identifier: Apache-2.0

//===----- MQHigh.td -- MQHigh Dialect Operation Definitions -*- tablegen ----==//
//
// Copyright 2024 The IBM Research Authors
//
// =============================================================================
//
// Defines MQHigh Dialect operations.
//
// When updating, run `make onnx-mlir-docs` to update the docs.
//
//===----------------------------------------------------------------------===//

#ifndef MQHIGH_OPS
#define MQHIGH_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/IR/AttrBase.td"

//===----------------------------------------------------------------------===//
// MQHigh Dialect
//===----------------------------------------------------------------------===//

def MQHigh_Dialect : Dialect {
  let name = "mqhigh";
  let summary = "A high-level dialect for the ONNX MQ accelerator ISA.";
  let cppNamespace = "::onnx_mlir::mqhigh";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// MQHigh Attribute and Type
//===----------------------------------------------------------------------===//
// TODO

//===----------------------------------------------------------------------===//
// MQHigh Operations
//===----------------------------------------------------------------------===//
// Base class for MQHigh dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.

class MQHigh_Op<string mnemonic, list<Trait> traits = []> :
    Op<MQHigh_Dialect, mnemonic, traits>;

// Conv2D Operation

// Elementwise Operations

// MatMul Operation
def MQHighMatMulOp:MQHigh_Op<"MatMul", [Pure
    // DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<ShapeHelperOpInterface>
    ]> {
  let summary = "MQHigh MatMul operation";
  let description = [{
    MQHigh operation to perform a MatMul.
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    Optional<AnyTensor>:$bias
  );

  let results = (outs
    AnyTensor:$result
  );

  let hasVerifier = 1;
}

// Pooling Operations

// Softmax Operation
// def ZHighSoftmaxOp:ZHigh_Op<"Softmax", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "ZHigh Softmax operation";
//   let description = [{
//     ZHigh operation to perform a Softmax.
//     act_func: ACT_NONE or ACT_LOG.
//   }];
//   let arguments = (ins ZTensor_3DS:$X,
//                        DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
//   let results = (outs ZTensor_3DS:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * ZHighSoftmaxOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

#endif // MQHIGH_OPS
