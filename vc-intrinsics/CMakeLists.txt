# CMPLRLLVM-24605: This is stub file to remove dependency vc-intrinsics repository
# during getting environment for running in-tree tests on deployed DPC++ compiler
# and runtime. 
# Directory containing this file should be specified in cmake parameter:
#     -DLLVMGenXIntrinsics_SOURCE_DIR=path
file(TOUCH dummy.cpp)

add_llvm_library(LLVMGenXIntrinsics
    dummy.cpp
)
