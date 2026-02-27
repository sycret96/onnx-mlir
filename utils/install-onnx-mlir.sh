LLVM_PROJECT_DIR=$(pwd)/../../llvm-project
INSTALL_DIR=$(pwd)/../../install

MLIR_DIR=${LLVM_PROJECT_DIR}/build/lib/cmake/mlir
LLVM_DIR=${LLVM_PROJECT_DIR}/build/lib/cmake/llvm

echo "MLIR_DIR: ${MLIR_DIR}"
echo "LLVM_DIR: ${LLVM_DIR}"
echo "INSTALL_DIR: ${INSTALL_DIR}"

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
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..

ninja 

ninja install

# # Run lit tests:
# export LIT_OPTS=-v
# cmake --build . --target check-onnx-lit
