#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/MQ/Compiler/MQCompilerUtils.hpp"
#include "src/Accelerators/MQ/MQAccelerator.hpp"

#include <memory>

#define DEBUG_TYPE "MQAccelerator"

namespace onnx_mlir {
namespace accel {

Accelerator *createMQ() { return MQAccelerator::getInstance(); }

MQAccelerator *MQAccelerator::instance = nullptr;

MQAccelerator *MQAccelerator::getInstance() {
  if (instance == nullptr)
    instance = new MQAccelerator();
  return instance;
}

MQAccelerator::MQAccelerator() : Accelerator(Kind::MQ) {
  // 必须把当前实例放入基类的静态向量中，否则 getAccelerators() 找不到它
  acceleratorTargets.push_back(this);
  this->setName("MQ"); // 设置加速器名称
}

MQAccelerator::~MQAccelerator() { delete instance; }

uint64_t MQAccelerator::getVersionNumber() const {
  return 1;
}

// --- Driver Hooks ---

void MQAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) const {
  // 以后在这里添加 MQ 特有的转换流水线
  llvm::outs() << "MQ Accelerator passes are being added!\n";
}

void MQAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  // 注册 MQ 的 Dialect，例如：registry.insert<mq::MQDialect>();
}

void MQAccelerator::registerPasses(int optLevel) const {
  // 注册命令行可用的 Pass
}

void MQAccelerator::configurePasses() const {
  // 统一配置 Pass 的参数
}

// --- Lowering Hooks ---

mlir::MemRefType MQAccelerator::convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const {
  // 如果 MQ 没有特殊的 Tensor 编码（Encoding），直接返回 nullptr
  // 这会让编译器使用标准的转换逻辑
  return nullptr;
}

int64_t MQAccelerator::getDefaultAllocAlignment(const mlir::TensorType type) const {
  // 如果MQ的 RISC-V 硬件有特定的对齐要求（如 64 字节），在此修改
  return -1;
}

void MQAccelerator::conversionTargetONNXToKrnl(mlir::ConversionTarget &target) const {
  // 设置哪些 Dialect/Op 在转换到 Krnl 阶段是合法的
}

void MQAccelerator::rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const {
  // 填充从 ONNX 降低到 MQ 相关 Op 的模式
}

void MQAccelerator::conversionTargetKrnlToLLVM(mlir::ConversionTarget &target) const {
  // 设置从 Krnl 降低到 LLVM 阶段的合法性目标
}

void MQAccelerator::rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) const {
  // 填充最终生成 LLVM IR 的模式
}

} // namespace accel
} // namespace onnx_mlir