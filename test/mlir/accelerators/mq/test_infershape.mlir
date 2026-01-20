func.func @test_infer_shape(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) {
  // 注意：这里我们故意写成 (tensor<128x64xf32>, tensor<64x32xf32>) -> <<MISSING>>
  // 预期就是会报错的 没关系
  %0 = "mqhigh.MatMul"(%arg0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<?x?xf32>
//   %0 = "mqhigh.MatMul"(%arg0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
  return
}