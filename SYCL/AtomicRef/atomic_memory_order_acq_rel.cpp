// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and ROCm backends don't currently support
// info::device::atomic_memory_order_capabilities
// UNSUPPORTED: level_zero || opencl || rocm

// NOTE: Tests load and store for acquire-release memory ordering.

#include "atomic_memory_order_acq_rel.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  if (!is_supported(supported_memory_orders, memory_order::acq_rel)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;

  // Acquire-release memory order must also support both acquire and release
  // orderings.
  assert(is_supported(supported_memory_orders, memory_order::acquire) &&
         is_supported(supported_memory_orders, memory_order::release));
  acq_rel_test<int>(q, N);
  acq_rel_test<unsigned int>(q, N);
  acq_rel_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    acq_rel_test<long>(q, N);
    acq_rel_test<unsigned long>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
