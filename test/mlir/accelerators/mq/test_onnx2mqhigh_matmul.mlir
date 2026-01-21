func.func @test_matmul(%arg0: tensor<32x16xf32>, %arg1: tensor<16x256xf32>) -> tensor<32x256xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x16xf32>, tensor<16x256xf32>) -> tensor<32x256xf32>
  return %0 : tensor<32x256xf32>
}