echo "=== Test 1: InferShape ==="
/home/lifan/workspace/onnx-mlir/build/Release/bin/onnx-mlir-opt --maccel=MQ ./test_infershape.mlir

echo ""
echo "=== Test 2: ONNX to MQHigh MatMul ==="
/home/lifan/workspace/onnx-mlir/build/Release/bin/onnx-mlir-opt --onnx-to-mqhigh --maccel=MQ ./test_onnx2mqhigh_matmul.mlir