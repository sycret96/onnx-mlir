# MLIR_DIR must be set with cmake option now
# MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
MLIR_DIR=/home/lifan/workspace/buddy-mlir/llvm/build/lib/cmake/mlir
LLVM_DIR=/home/lifan/workspace/buddy-mlir/llvm/build/lib/cmake/llvm

mkdir -p build
cd build
cmake -G Ninja \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DONNX_MLIR_ENABLE_STABLEHLO=OFF \
      -DMLIR_DIR=${MLIR_DIR} \
      -DLLVM_DIR=${LLVM_DIR} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..

cmake --build . 



# mkdir onnx-mlir/build && cd onnx-mlir/build
# if [[ -z "$pythonLocation" ]]; then
#   cmake -G Ninja \
#         -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DLLVM_ENABLE_ASSERTIONS=ON \
#         -DMLIR_DIR=${MLIR_DIR} \
#         ..
# else
#   cmake -G Ninja \
#         -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DLLVM_ENABLE_ASSERTIONS=ON \
#         -DPython3_ROOT_DIR=$pythonLocation \
#         -DMLIR_DIR=${MLIR_DIR} \
#         ..
# fi
# cmake --build .

# # Run lit tests:
# export LIT_OPTS=-v
# cmake --build . --target check-onnx-lit
