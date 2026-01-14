# MLIR_DIR must be set with cmake option now
export CMAKE_PREFIX_PATH=$CONDA_PREFIX

MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
mkdir -p build
cd build
cmake -G Ninja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DONNX_MLIR_ENABLE_STABLEHLO=OFF \
      -DMLIR_DIR=${MLIR_DIR} \
      ..

cmake --build .
# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
