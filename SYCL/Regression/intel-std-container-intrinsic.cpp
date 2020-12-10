// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// The purpose of this test is to check that we are able to successfully compile
// the following example:
// TODO: we might want to consider updating our optimization pipeline in order
// to make this test passing even without early optimizations enabled:
// clang++ -fsycl -fno-sycl-early-optimizations command will fail on this
// example with the following error:
// InvalidFunctionCall: Unexpected llvm intrinsic:
// llvm.intel.std.container.ptr.p4i32 [Src: llvm-spirv/lib/SPIRV/SPIRVWriter.cpp:2668  ]
// llvm-foreach:
// clang-12: error: llvm-spirv command failed with exit code 1 (use -v to see invocation)

#include <CL/sycl.hpp>
#include <vector>

using namespace cl::sycl;

using usm_int_allocator = usm_allocator<int, usm::alloc::shared>;
using usm_vec_allocator =
    usm_allocator<std::vector<int, usm_int_allocator>, usm::alloc::shared>;

int main() {
  queue q;
  usm_int_allocator alloc(q);
  usm_vec_allocator valloc(q);
  std::vector<int, usm_int_allocator> *vmem = valloc.allocate(1);
  new (vmem) std::vector<int, usm_int_allocator>(1024, 0, alloc);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class init_a>(
        range<1>{1024}, [=](id<1> index) { (*vmem)[index[0]] = index[0]; });
  });
}
