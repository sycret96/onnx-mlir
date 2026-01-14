// RUN: onnx-mlir-opt %s --split-input-file | FileCheck %s

// 这是一个简单的冒烟测试，确保 MoonQuest Dialect 能够被解析
// CHECK: module
module {
  func.func @test_presence() {
    // 这里可以先不写任何你自定义的 Op
    // 只要能通过编译和基本解析即可
    return
  }
}