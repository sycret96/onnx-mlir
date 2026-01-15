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
// MQHigh Attribute
//===----------------------------------------------------------------------===//
// TODO

//===----------------------------------------------------------------------===//
// MQHigh Operations
//===----------------------------------------------------------------------===//

class MQHigh_Op<string mnemonic, list<Trait> traits = []> :
    Op<MQHigh_Dialect, mnemonic, traits>;

#endif // MQHIGH_OPS
