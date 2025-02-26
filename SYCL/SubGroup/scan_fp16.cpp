// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test verifies the correct work of the sub-group algorithms
// exclusive_scan() and inclusive_scan().

#include "scan.hpp"

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device()) ||
      !Queue.get_device().has_extension("cl_khr_fp16")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<class KernelName_dlpo, cl::sycl::half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
