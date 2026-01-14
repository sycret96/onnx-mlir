#ifndef ONNX_MLIR_MQ_ACCELERATOR_H
#define ONNX_MLIR_MQ_ACCELERATOR_H

#include "mlir/IR/BuiltinTypes.h"
#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

/// Singleton class to construct an MQ accelerator.
class MQAccelerator final : public Accelerator {
private:
  static MQAccelerator *instance;
  MQAccelerator();

public:
  /// Singleton should not be clonable or assignable.
  MQAccelerator(MQAccelerator &) = delete;
  void operator=(const MQAccelerator &) = delete;

  ~MQAccelerator();

  /// Creates an instance on the first invocation. Subsequent invocations
  /// return the existing instance.
  static MQAccelerator *getInstance();

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const Accelerator *accel) {
    return accel->getKind() == Accelerator::Kind::MQ;
  }
  static bool classof(const MQAccelerator *) { return true; }

  uint64_t getVersionNumber() const final;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir driver
  //===--------------------------------------------------------------------===//
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
      std::string outputNameNoExt) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//
  virtual void registerDialects(mlir::DialectRegistry &registry) const final;
  virtual void registerPasses(int optLevel) const final;
  virtual void configurePasses() const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-to-krnl pass
  //===--------------------------------------------------------------------===//
  virtual mlir::MemRefType convertTensorTypeToMemRefType(
      const mlir::TensorType tensorType) const final;
  virtual void conversionTargetONNXToKrnl(
      mlir::ConversionTarget &target) const final;
  virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
      mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const final;
  virtual int64_t getDefaultAllocAlignment(
      const mlir::TensorType tensorType) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for krnl-to-llvm pass
  //===--------------------------------------------------------------------===//
  virtual void conversionTargetKrnlToLLVM(
      mlir::ConversionTarget &target) const final;
  virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
      mlir::LLVMTypeConverter &typeConverter,
      mlir::MLIRContext *ctx) const final;
};

} // namespace accel
} // namespace onnx_mlir
#endif // ONNX_MLIR_MQ_ACCELERATOR_H